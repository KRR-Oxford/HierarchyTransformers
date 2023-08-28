import torch

torch.set_default_dtype(torch.float64)
from hypertrans.graph import HypernymGraph
from hypertrans import PoincareTrainer
import click

data_choices = {
    "wordnet": "./data/wordnet/wordnet_hypernyms.tsv",
    "mammal": "./data/wordnet/wordnet_mammal_hypernyms.tsv",
}


@click.command()
@click.option("--data", type=click.Choice(["wordnet", "mammal"]), required=True)
@click.option("--embed_dim", type=int, default=100)
@click.option("--num_negative_samples", type=int, default=10)
@click.option("--batch_size", type=int, default=512)
@click.option("--num_epochs", type=int, default=50)
@click.option("--num_warmup_epochs", type=int, default=10)
@click.option("--device", type=int, default=0)
def main(data, embed_dim, num_negative_samples, batch_size, num_epochs, num_warmup_epochs, device):
    graph = HypernymGraph(data_choices[data])
    trainer = PoincareTrainer(
        graph,
        embed_dim=embed_dim,
        n_negative_samples=num_negative_samples,
        batch_size=batch_size,
        n_epochs=num_epochs,
        n_warmup_epochs=num_warmup_epochs,
        gpu_device=device,
    )
    trainer.run()
    torch.save(
        trainer.model,
        f"experiments/poincare.{data}.dim={embed_dim}.neg={num_negative_samples}.batch={batch_size}.epoch={num_epochs}.warmup={num_warmup_epochs}.pt",
    )

main()
