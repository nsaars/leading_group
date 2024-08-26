from llama_parse import LlamaParse

# Initialize the LlamaParse parser with your API key and desired output format
parser = LlamaParse(
    api_key="llx-bNkvGxMlnN9Ft3TNvLDkZmLxiBIftCbxsUdeUliFhRkUUFnT",  # This can also be set as an environment variable named LLAMA_CLOUD_API_KEY
    result_type="markdown",  # Choose between "markdown" and "text" output formats
    verbose=True  # Enables detailed logging for debugging purposes
)

