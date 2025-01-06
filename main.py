import hashlib
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print
from typing_extensions import Annotated

from ragthoven import Ragthoven
from ragthoven.executors.output_writer import (
    CSVOutputWriter,
    JSONLOutputWriter,
    SupportedOutputFormats,
)
from ragthoven.models.base import Config
from ragthoven.models.iter_matrix import IterationMatrix
from ragthoven.utils import stringify_obj, stringify_obj_beautiful
from ragthoven.utils.config_loader import load_config

app = typer.Typer()

# Load (override defaults) environment variables from .env file
load_dotenv(override=True)


@app.command()
def rag(
    config_path: str,
    output_format: Annotated[
        Optional[SupportedOutputFormats], typer.Option("--output")
    ] = None,
):
    print("Starting RAG ...")

    config = load_config(config_path)
    iter_matrix = IterationMatrix(config)

    cont = True
    while cont:
        iter_matrix.build_config()
        current_config: Config = iter_matrix.get_config()

        # Process terminal options
        output_writer = None
        output_base_name = current_config.results.output_filename

        # write metadata
        object_stringified = stringify_obj(current_config)
        config_hash = hashlib.sha256(object_stringified.encode())
        hex_dig = str(config_hash.hexdigest())[:12]
        with open(f"{output_base_name}.{hex_dig}.metadata.json", "w") as metadata_file:
            metadata_file.write(stringify_obj_beautiful(current_config))

        if output_format == SupportedOutputFormats.JSONL:
            filename = (
                f"{output_base_name}.{hex_dig}.{SupportedOutputFormats.JSONL.value}"
            )
            output_writer = JSONLOutputWriter(filename, current_config)
        elif output_format == SupportedOutputFormats.CSV:
            filename = (
                f"{output_base_name}.{hex_dig}.{SupportedOutputFormats.CSV.value}"
            )
            output_writer = CSVOutputWriter(filename, current_config)
        else:  # fallback to csv
            filename = (
                f"{output_base_name}.{hex_dig}.{SupportedOutputFormats.JSONL.value}"
            )
            output_writer = JSONLOutputWriter(filename, current_config)

        r = ragthoven(current_config, output_write=output_writer)
        r.execute()
        cont = iter_matrix.inc()

    print("RAG done!")


if __name__ == "__main__":
    app()
