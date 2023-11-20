from deeponto.onto import Taxonomy
from datasets import Dataset
from sentence_transformers import LoggingHandler, InputExample


from tqdm.auto import tqdm


def example_generator(taxonomy: Taxonomy, dataset: Dataset):
    examples = []
    for sample in tqdm(dataset, leave=True, desc=f"Prepare examples for {dataset.split._name}"):
        child = taxonomy.get_node_attributes(sample["child"])["name"]
        parent = taxonomy.get_node_attributes(sample["parent"])["name"]
        negative_parents = [taxonomy.get_node_attributes(neg)["name"] for neg in sample["negative_parents"]]
        examples.append(InputExample(texts=[child, parent], label=1))
        examples += [InputExample(texts=[child, neg], label=0) for neg in negative_parents]
    return examples
