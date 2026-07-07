import torch
import numpy as np
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers.trainer_utils import EvalPrediction
from utils.tools import backbone_selection
from models.TS_model import TS_model
from datasets_split.datasets_split import Datasets_split
from utils.metrics import metric
from torch.utils.data import DataLoader


def custom_collate_fn(batch):
    seq_x = torch.stack([item[0] for item in batch])
    seq_y = torch.stack([item[1] for item in batch])
    return seq_x, seq_y

def custom_metric(pred, true):
    mae, mse, rmse, mape, mspe, smape, nd = metric(pred, true)
    return {
        "MAE": mae,
        "MSE": mse
    }


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred
    preds = predictions.astype(np.float32)
    labels = labels.astype(np.float32)
    return custom_metric(preds, labels)


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        batch_x = inputs[0]
        batch_y = inputs[1]
        batch_x = batch_x.unsqueeze(-1)
        batch_y = batch_y.unsqueeze(-1)

        outputs = model(batch_x)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(outputs, batch_y[:, -self.args.pred_len:, :])
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            drop_last=False,
            collate_fn=custom_collate_fn,
        )
        return self.accelerator.prepare(train_dataloader)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        val_dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            drop_last=False,
            collate_fn=custom_collate_fn
        )
        return self.accelerator.prepare(val_dataloader)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        batch_x, batch_y = inputs
        batch_x = batch_x.unsqueeze(-1)
        batch_y = batch_y.unsqueeze(-1)

        with torch.no_grad():
            outputs = model(batch_x)
            labels = batch_y[:, -self.args.pred_len:, :]
        return None, outputs, labels


def model_infer(dataset_tr, dataset_va, dataset_te, test_metrics_path, args):
    args = backbone_selection(args)
    model = TS_model(args)

    train_dataset = Datasets_split(size=[args.seq_len, args.pred_len], dataset=dataset_tr)
    valid_dataset = Datasets_split(size=[args.seq_len, args.pred_len], dataset=dataset_va)
    test_dataset = Datasets_split(size=[args.seq_len, args.pred_len], dataset=dataset_te)

    training_args = TrainingArguments(
        ddp_find_unused_parameters=False,
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        learning_rate=5e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=512,
        seed=2025,
        dataloader_num_workers=20,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        num_train_epochs=4 if args.cross_datasets else 30,
        eval_strategy="epoch" if args.is_save_checkpoint else "no",
        save_strategy="epoch" if args.is_save_checkpoint else "no",
        load_best_model_at_end=True if args.is_save_checkpoint else False,
        metric_for_best_model="eval_MSE",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        push_to_hub=False,
        bf16=True,
        gradient_accumulation_steps=1,
        save_total_limit=6,
        optim="adamw_torch",
        save_only_model=False,
        save_safetensors=False,
        report_to=None,
        pred_len=args.pred_len,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer._load_from_checkpoint(checkpoint_path) # load checkpoint_path
    results = trainer.evaluate(valid_dataset) # Use the validation set for analysis to prevent data leakage from the test set during the analysis process.
    model.save_all_attentions(f"main/single_dataset_learning/inference/all_attentions.pt")

    torch.cuda.empty_cache()