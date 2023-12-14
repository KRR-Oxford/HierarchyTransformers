import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import click
from yacs.config import CfgNode
from deeponto.utils import load_file, save_file
import logging

logger = logging.getLogger(__name__)

from sentence_transformers import SentenceTransformer
from hierarchy_transformers.utils import *
from hierarchy_transformers.evaluation import *
from hierarchy_transformers.models.hit import get_circum_poincareball


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-g", "--gpu_id", type=int, default=0)
def main(config_file: str, gpu_id: int):
    # set_seed(8888)
    config = CfgNode(load_file(config_file))

    data_path = config.data_path
    dataset, entity_lexicon = load_hierarchy_dataset(data_path, eval_only=True)
    prepare_func = prepare_hierarchy_examples
    if config.model_type == "finetune":
        prepare_func = prepare_hierarchy_examples_for_finetune
    trans_val_examples = prepare_func(entity_lexicon, dataset["transitivity"]["val"], config.apply_hard_negatives)
    trans_test_examples = prepare_func(entity_lexicon, dataset["transitivity"]["test"], config.apply_hard_negatives)
    base_val_examples = prepare_func(entity_lexicon, dataset["completion"]["val"], config.apply_hard_negatives)
    base_test_examples = prepare_func(entity_lexicon, dataset["completion"]["test"], config.apply_hard_negatives)

    if config.model_type == "hit":
        device = get_torch_device(gpu_id)
        model = SentenceTransformer(
            config.pretrained,
            device=device,
        )
        manifold = get_circum_poincareball(model._first_module().get_word_embedding_dimension())

        def hit_result_mat(examples):
            child_embeds = model.encode([x.texts[0] for x in examples], config.eval_batch_size, False, convert_to_tensor=True)
            parent_embeds = model.encode([x.texts[1] for x in examples], config.eval_batch_size, False, convert_to_tensor=True)
            labels = torch.tensor([x.label for x in examples]).to(child_embeds.device)
            dists = manifold.dist(child_embeds, parent_embeds)
            child_norms = manifold.dist0(child_embeds)
            parent_norms = manifold.dist0(parent_embeds)
            return torch.stack([labels, dists, child_norms, parent_norms]).T
            # return dists + centri_score_weight * (parent_norms - child_norms), labels
        
        trans_val_result_mat = hit_result_mat(trans_val_examples)
        trans_val_results = HierarchyRetrainedEvaluator.search_best_threshold(trans_val_result_mat)
        trans_test_result_mat = hit_result_mat(trans_test_examples)
        trans_test_scores = trans_test_result_mat[:, 1] + trans_val_results["centri_score_weight"] * (trans_test_result_mat[:, 3] - trans_test_result_mat[:, 2])
        trans_test_results = evaluate_by_threshold(trans_test_scores, trans_test_result_mat[:, 0], trans_val_results["threshold"])
        save_file(trans_test_results, f"{config.pretrained}/transfer_trans_test_results.hard={config.apply_hard_negatives}.json")

        base_val_result_mat = hit_result_mat(base_val_examples)
        base_val_results = HierarchyRetrainedEvaluator.search_best_threshold(base_val_result_mat)
        base_test_result_mat = hit_result_mat(base_test_examples)
        base_test_scores = base_test_result_mat[:, 1] + base_val_results["centri_score_weight"] * (base_test_result_mat[:, 3] - base_test_result_mat[:, 2])
        base_test_results = evaluate_by_threshold(base_test_scores, base_test_result_mat[:, 0], base_val_results["threshold"])
        save_file(base_test_results, f"{config.pretrained}/transfer_base_test_results.hard={config.apply_hard_negatives}.json")

    elif config.model_type == "finetune":
        model = AutoModelForSequenceClassification.from_pretrained(config.pretrained)
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained)
        tok_func = lambda example: tokenizer(example["entity1"], example["entity2"], truncation=True, max_length=256)
        train_args = TrainingArguments(".", per_device_eval_batch_size=config.eval_batch_size)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(model, train_args, data_collator=data_collator, tokenizer=tokenizer)

        trans_test_examples = Dataset.from_list(trans_test_examples).map(tok_func, batched=True)
        trans_test_preds = trainer.predict(trans_test_examples)
        trans_test_scores = torch.tensor(trans_test_preds.predictions).argmax(dim=1)
        trans_test_labels = torch.tensor(trans_test_preds.label_ids)
        trans_test_results = evaluate_by_threshold(trans_test_scores, trans_test_labels, 0.0, False)
        save_file(trans_test_results, f"{config.pretrained}/../transfer_trans_test_results.hard={config.apply_hard_negatives}.json")

        base_test_examples = Dataset.from_list(base_test_examples).map(tok_func, batched=True)
        base_test_preds = trainer.predict(base_test_examples)
        base_test_scores = torch.tensor(base_test_preds.predictions).argmax(dim=1)
        base_test_labels = torch.tensor(base_test_preds.label_ids)
        base_test_results = evaluate_by_threshold(base_test_scores, base_test_labels, 0.0, False)
        save_file(base_test_results, f"{config.pretrained}/../transfer_base_test_results.hard={config.apply_hard_negatives}.json")


if __name__ == "__main__":
    main()
