import os
import logging
import argparse
import yaml  # You'll need to install this: pip install pyyaml
import warnings
import time
# Import the necessary functions from your NeMo utility files
from utils.setup_logger import setup_logger
from utils.set_device import get_device_type_index
from asr_utils.nemo_pipeline import process_transcription, process_transcription_function, validate_audio_file
from asr_utils.nemo_class import NeMo_Transcriber


# ========== Global Variables ==========
base_dir = os.path.dirname(os.path.abspath(__file__))
# Updated logger names for the NeMo pipeline
original_log_names = [
   'utils.set_device',
   'utils.load_audio',
   'asr_utils.nemo_pipeline',
   'asr_utils.nemo_class',
   'nemo_logger',          
   'py.warnings',
   'initialization',
   'nemo_pipeline.process_transcription_function',
   'nemo_pipeline.process_transcription'
]
# ========== Disable NeMo Logger ==========
def disable_nemo_logger_handlers():
    nemo_logger = logging.getLogger("nemo_logger")
    if nemo_logger.hasHandlers():
        nemo_logger.handlers.clear()
    nemo_logger.propagate = False
# ========== Main Functions ==========
def nemo_chunking_transcription(args):
    """
    Main function to handle the NeMo transcription workflow.
    """
    if args.do_logging:
        disable_nemo_logger_handlers()
        setup_logger(original_log_names, args.log_path, args.log_level, args.use_log_file, args.log_use_colors)
    else:
        # If logging is disabled, also suppress any warnings and the progress bar
        warnings.filterwarnings('ignore')
    
    if not validate_audio_file(args.audio_filepath):
        return

    device_type, device_index = get_device_type_index(args.device)
    # Instantiate the NeMo transcriber directly, replacing the initialize_models function
    logger = logging.getLogger("initialization")
    logger.info("Initializing NeMo ASR model...")
    start_time_asr_init = time.time()
    transcriber = NeMo_Transcriber(
        model_path=args.model_path,
        decoder_type=args.decoder_type,
        device_type=device_type,
        device_index=device_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
        extract_nbest=args.extract_nbest
    )
    end_time_asr_init = time.time()
    logger.info("Model initialized successfully.")
    logger.info(f"ASR model initialization took {end_time_asr_init - start_time_asr_init:.2f} seconds.")
    if args.save_nothing:
        text, segments = process_transcription_function(
            audio_filepath=args.audio_filepath,
            transcriber=transcriber,
            sampling_rate=args.sampling_rate,
            chunk_length=args.chunk_length,
            text_normalization=args.text_normalization,
            text_post_normalization=args.text_post_normalization,
            save_only_final_transcript=args.save_only_final_transcript)
        # Print the results to the console
        if text is not None:
            print("\n--- TRANSCRIPTION RESULT ---")
            print(text.strip())
            if segments:
                print(f"\n(Generated {len(segments)} segments)")
        else:
            print("\n--- TRANSCRIPTION FAILED ---")

    else:      
        # Call the processing function from your NeMo pipeline script
        text, segments = process_transcription(
            audio_filepath=args.audio_filepath,
            model_path=args.model_path,
            received_files_path=args.received_files_path,
            transcriber=transcriber,
            chunk_length=args.chunk_length,
            sampling_rate=args.sampling_rate,
            save_segments=args.save_segments,
            text_normalization=args.text_normalization,
            text_post_normalization=args.text_post_normalization,
            save_only_final_transcript=args.save_only_final_transcript
        )
        if text is not None:
            print("\n--- TRANSCRIPTION COMPLETE ---")
            print(f"Outputs have been saved to the designated folder.")
            print(f"Found {len(segments)} segments.")
        else:
            print("\n--- TRANSCRIPTION FAILED ---")

def main():
    parser = argparse.ArgumentParser(
        description="Process an audio file using NeMo ASR with fixed-length chunking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-c", "--config", type=str,
        help="Path to a YAML configuration file to set arguments."
    )

    # --- Group 1: File and Path Arguments ---
    path_args = parser.add_argument_group("File and Path Arguments")
    path_args.add_argument("audio_filepath", type=str, nargs='?', default=None, help="Path to the input audio file.")
    path_args.add_argument("--received_files_path", type=str, default=None, help="Path to the output folder. Not required if --save-nothing is used.")
    path_args.add_argument("--model-path", type=str, help="Path to the NeMo ASR model.")
    path_args.add_argument("--log-path", type=str, help="Path to the log file.")




    # --- Group 2: Model, Device & Audio Arguments ---
    model_device_args = parser.add_argument_group("Model, Device & Audio Arguments")
    model_device_args.add_argument("--device", type=str, default="cpu", help="Device for ASR model (e.g., 'cuda', 'cpu').")
    model_device_args.add_argument("--use-amp", action="store_true", help="Enable Automatic Mixed Precision for faster inference.")
    model_device_args.add_argument("--sampling_rate", type=int, default=16000, help="Audio sampling rate required by the model.")
    model_device_args.add_argument("--num_workers", type=int, default=0, help="Number of workers for the dataloader.")
    model_device_args.add_argument('--channel-selection', dest='channel_selection', type=str, default='average', choices=['average', 'left', 'right'], help="Select which audio channel to process.")


    # --- Group 3: Transcription Control Arguments ---
    transcription_args = parser.add_argument_group("Transcription Control Arguments")
    transcription_args.add_argument("--batch_size", type=int, default=32, help="Batch size for transcription.")
    transcription_args.add_argument("--chunk-length", type=float, default=60.0, help="Length of audio chunks in seconds for processing long files.")
    transcription_args.add_argument("--decoder-type", type=str, default="rnnt", choices=["rnnt", "ctc"], help="Type of decoder to use if the model supports it.")
    transcription_args.add_argument("--extract-nbest", action="store_true", help="Extract n-best hypotheses instead of only the best one.")


    # --- Group 4: Text Processing & Output Arguments ---
    output_args = parser.add_argument_group("Text Processing & Output Arguments")
    output_args.add_argument('--text-normalization', action='store_true', help="Enable initial text normalization stage.")
    output_args.add_argument("--no-text-post-normalization", dest="text_post_normalization", action="store_false", help="Disable post-normalization step.")
    output_args.add_argument('--no-save-only-final-transcript', action='store_false', dest='save_only_final_transcript', help="Keep all text versions in the final JSON output.")
    output_args.add_argument('--no-save-segments', action='store_false', dest='save_segments', help="Disable saving of individual audio/text segments.")
    output_args.add_argument('--save-nothing', action='store_true', help="Run transcription in memory and print the final text without saving any files.")


    # --- Group 5: Logging Arguments ---
    log_args = parser.add_argument_group("Logging Arguments")
    log_args.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Set the logging level.")
    log_args.add_argument('--no-log-progress', action='store_false', dest='log_progress', help="Disable the transcription progress bar.")
    log_args.add_argument("--no-color", dest="log_use_colors", action="store_false", help="Disable ANSI colors in console logs.")
    log_args.add_argument("--no-log-file", dest="use_log_file", action="store_false", help="Disable logging to a file.")
    log_args.add_argument('--no-logging', action='store_false', dest='do_logging', help="Disable all logging output.")


    # First, do a partial parse to find the config file path
    temp_args, _ = parser.parse_known_args()

    # If a config file is provided, load it and set the values as defaults
    if temp_args.config and os.path.exists(temp_args.config):
        print(f"Loading configuration from: {temp_args.config}")
        with open(temp_args.config, 'r') as f:
            config_from_file = yaml.safe_load(f)
        flat_config = {}
        if config_from_file:
            # Flatten the YAML structure into a single dictionary
            for group_values in config_from_file.values():
                if isinstance(group_values, dict):
                    flat_config.update(group_values)
            # Set these values as the new defaults
            parser.set_defaults(**flat_config)

    # Now, parse arguments a final time. Command-line args will override config file values.
    args = parser.parse_args()
    
    # Check if audio_filepath was provided either on CLI or in config
    if not args.audio_filepath:
        parser.error("the following arguments are required: audio_filepath")


    # --- Set default paths if not provided ---
    if not args.save_nothing and not args.received_files_path:
        args.received_files_path = os.path.join(base_dir, "Received_Files_CLI")
    if not args.log_path:
        args.log_path = os.path.join(base_dir, "CLI_Logs", "nemo_transcriber.log")
    if not args.model_path:
        args.model_path = os.path.join(base_dir, "models/asr/nemo-stt-fastconformer-hybrid-large-v40/model-epoch_34-step_170869-val_wer_0.08.nemo")

    nemo_chunking_transcription(args)

if __name__ == "__main__":
    main()