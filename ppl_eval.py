from hite.evaluation import MaskFillPerplexityEvaluator
from hite.utils import get_device, load_hierarchy_dataset, example_generator

peval = MaskFillPerplexityEvaluator("sentence-transformers/all-MiniLM-L6-v2", get_device(1))

dataset, entity_lexicon = load_hierarchy_dataset("data/wordnet")

val_examples = example_generator(entity_lexicon, dataset["trans"]["val"])
test_examples = example_generator(entity_lexicon, dataset["trans"]["test"])
peval(val_examples, test_examples, "experiments/wordnet/perplexity/trans/random")

val_examples = example_generator(entity_lexicon, dataset["trans"]["val"], True)
test_examples = example_generator(entity_lexicon, dataset["trans"]["test"], True)
peval(val_examples, test_examples, "experiments/wordnet/perplexity/trans/hard")

val_examples = example_generator(entity_lexicon, dataset["induc"]["val"])
test_examples = example_generator(entity_lexicon, dataset["induc"]["test"])
peval(val_examples, test_examples, "experiments/wordnet/perplexity/induc/random")

val_examples = example_generator(entity_lexicon, dataset["induc"]["val"], True)
test_examples = example_generator(entity_lexicon, dataset["induc"]["test"], True)
peval(val_examples, test_examples, "experiments/wordnet/perplexity/induc/hard")
