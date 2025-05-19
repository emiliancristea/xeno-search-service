import asyncio
from transformers import pipeline, set_seed
from typing import Optional
import logging
import torch # Ensure torch is imported
import functools # Add this import

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the summarization pipeline globally but lazily
# This helps in not loading the model at import time unless it's actually used.
# We'll use a placeholder and load it on first use.
summarizer_pipeline = None
summarizer_model_name = "sshleifer/distilbart-cnn-12-6" # Changed model

def get_summarizer():
    global summarizer_pipeline
    if summarizer_pipeline is None:
        try:
            logger.info(f"Initializing summarization pipeline with model: {summarizer_model_name}...")
            # Attempt to use CUDA if available
            device = 0 if torch.cuda.is_available() else -1
            logger.info(f"Summarization pipeline will attempt to use device: {'cuda:0' if device == 0 else 'cpu'}")
            summarizer_pipeline = pipeline(
                "summarization",
                model=summarizer_model_name,
                tokenizer=summarizer_model_name,
                device=device # Use GPU if available, otherwise CPU
            )
            logger.info("Summarization pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize summarization pipeline: {e}")
            # Potentially raise the error or handle it by setting pipeline to a non-functional state
            raise
    return summarizer_pipeline

async def summarize_text_async(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    do_sample: bool = False # BART default is False, T5 might be True
) -> Optional[str]:
    """
    Asynchronously summarizes the given text using a pre-trained model.
    The actual model inference is run in a separate thread to avoid blocking asyncio loop.
    """
    if not text or not text.strip():
        logger.warning("Input text is empty or whitespace only. Skipping summarization.")
        return None

    try:
        # Get the pipeline (it will be initialized on first call)
        summarizer = get_summarizer()
        if summarizer is None:
            logger.error("Summarizer pipeline is not available. Cannot summarize.")
            return None

        # The pipeline itself might be blocking, so run it in a thread
        loop = asyncio.get_event_loop()

        # Pre-process text: Ensure it's not excessively long for the model
        # BART model max tokens is 1024. 5000 chars is a safer upper bound for char truncation.
        max_input_char_length = 5000 # Reduced from 10000
        if len(text) > max_input_char_length:
            logger.warning(f"Input text exceeds {max_input_char_length} chars. Truncating for summarization.")
            text_to_summarize = text[:max_input_char_length]
        else:
            text_to_summarize = text

        # Ensure text_to_summarize is still valid after potential truncation
        if not text_to_summarize or not text_to_summarize.strip():
            logger.warning("Text became empty or whitespace after character truncation. Skipping summarization.")
            return None

        logger.info(f"Starting summarization for text of length {len(text_to_summarize)} characters.")
        
        # Wrap the summarizer call with its arguments
        summarizer_call = functools.partial(
            summarizer,
            text_to_summarize,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            truncation=True # Ensure tokenizer truncates to model's max input length
        )
        
        # Run the blocking pipeline call in a separate thread
        summary_list = await loop.run_in_executor(
            None,  # Uses the default ThreadPoolExecutor
            summarizer_call # Pass the wrapped call
        )

        if summary_list and isinstance(summary_list, list) and len(summary_list) > 0:
            summary = summary_list[0]['summary_text']
            logger.info(f"Summarization successful. Summary length: {len(summary)}")
            return summary
        else:
            logger.warning("Summarization did not return expected output.")
            return None
    except IndexError as e:
        logger.error(f"IndexError during summarization: {e}. Problematic text snippet (first 200 chars): '{text_to_summarize[:200]}...'")
        logger.exception("Traceback for IndexError in summarization:")
        return "Error: Summarization failed due to an indexing issue with the input."
    except RuntimeError as e:
        if "generator raised StopIteration" in str(e):
            logger.warning(f"Summarization failed: Input text might be too short or unsuitable for the model. Error: {e}")
            return None # Or return a specific message like "Content too short to summarize"
        elif "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory during summarization. Try reducing input size or using a smaller model. Error: {e}")
            # Potentially try again on CPU if that's an option, or just fail
            return "Error: Summarization failed due to resource limits."
        else:
            logger.error(f"An unexpected runtime error occurred during summarization: {e}")
            return None # Or a generic error message
    except Exception as e:
        logger.error(f"An unexpected error occurred during summarization: {e}")
        # Log the full traceback for debugging
        import traceback
        logger.error(traceback.format_exc())
        return None # Or a generic error message

# Example usage (for testing this module directly)
if __name__ == "__main__":
    async def main():
        sample_text_short = "This is a very short text. It might be hard to summarize."
        sample_text_long = (
            "Jupiter is the fifth planet from the Sun and the largest in the Solar System. "
            "It is a gas giant with a mass more than two and a half times that of all the other planets in the Solar System combined, "
            "but slightly less than one-thousandth the mass of the Sun. Jupiter is the third brightest natural object in the Earth's night sky "
            "after the Moon and Venus. People have been observing it since prehistoric times; it was named after the Roman god Jupiter, "
            "the king of the gods, because of its observed size."
            "Jupiter is primarily composed of hydrogen, but helium constitutes one-quarter of its mass and one-tenth of its volume. "
            "It likely has a rocky core of heavier elements, but like the other giant planets, Jupiter lacks a well-defined solid surface. "
            "Because of its rapid rotation, the planet's shape is that of an oblate spheroid (it has a slight but noticeable bulge around the equator). "
            "The outer atmosphere is visibly segregated into several bands at different latitudes, resulting in turbulence and storms along their "
            "interacting boundaries. A prominent result is the Great Red Spot, a giant storm that is known to have existed since at least the 17th "
            "century when it was first seen by telescope."
        )

        print("\nTesting with short text:")
        summary_short = await summarize_text_async(sample_text_short, min_length=5, max_length=20)
        if summary_short:
            print(f"Original: {sample_text_short}")
            print(f"Summary: {summary_short}")
        else:
            print("Summarization failed or returned None for short text.")

        print("\nTesting with longer text:")
        summary_long = await summarize_text_async(sample_text_long)
        if summary_long:
            print(f"Original (first 100 chars): {sample_text_long[:100]}...")
            print(f"Summary: {summary_long}")
        else:
            print("Summarization failed or returned None for long text.")
            
        # Test with empty string
        print("\nTesting with empty text:")
        summary_empty = await summarize_text_async("")
        if summary_empty is None:
            print("Summarization correctly returned None for empty text.")
        else:
            print(f"Summarization returned for empty text: {summary_empty}")


    asyncio.run(main())
