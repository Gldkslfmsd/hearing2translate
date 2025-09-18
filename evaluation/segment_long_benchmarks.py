from mweralign.mweralign import *
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import argparse
import json
import logging
from pathlib import Path

PATH_TO_NLLB_TOKENIZER = ''

def load_jsonl(file_path: str):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except FileNotFoundError:
        logging.info(f"Error: File not found at {file_path}")
        return []
    except json.JSONDecodeError:
        logging.info(f"Error: Could not decode JSON in {file_path}")
        return []
    return data

def segment(refs, hyps, language, no_detok=False):

    segmenter = SPSegmenter(PATH_TO_NLLB_TOKENIZER)

    def tokenize_and_join(text: List[str]) -> List[str]:
        """Tokenize text using the segmenter."""
        if segmenter is not None:
            for i in range(len(text)):
                if " ### " in text[i]:
                    pieces = text[i].strip().split(" ### ")
                    text[i] = " ### ".join([" ".join(segmenter.encode(p)) for p in pieces])
                elif "\t" in text[i]:
                    pieces = text[i].strip().split("\t")
                    # underlying C++ binary still uses ###
                    text[i] = " ### ".join([" ".join(segmenter.encode(p)) for p in pieces])
                else:
                    text[i] = " ".join(segmenter.encode(text[i].strip()))
        return "\n".join(text)

    docids = []
    docids = ["0"] * len(refs)
    hyps = [" ".join(hyps)]

    if len(docids) != len(refs):
        logger.info(f"Error: Number of docids ({len(docids)}) does not match number of references ({len(refs)}).")

    # make sure the number of distinct docids matches the number of hypotheses
    if len(set(docids)) != len(hyps):
        logger.info(f"Error: Number of distinct docids ({len(set(docids))}) does not match number of hypotheses ({len(hyps)}).")

    # build a list of docid ranges
    current_docid_start = 0
    current_docid = docids[0]
    docid_ranges = []
    for i in range(1, len(docids)):
        if docids[i] != current_docid:
            docid_ranges.append((current_docid_start, i))
            current_docid_start = i
            current_docid = docids[i]
    if current_docid_start < len(docids):
        docid_ranges.append((current_docid_start, len(docids)))

    # This param causes the AS-WER algorithm to disallow internal tokens
    # at the start of sentences (via a high cost penalty). This is important
    # in whitespace languages, but is not what we want with C&J, where most tokens
    # appear to be internal because there was no whitespace.
    is_tokenized = type(segmenter) is SPSegmenter and language not in ["ja", "zh"]

    for i, (docid_start, docid_end) in enumerate(docid_ranges):
        hyp_str = tokenize_and_join([hyps[i]])
        ref_str = tokenize_and_join(refs[docid_start:docid_end])

        logger.info(f"Aligning {len(hyp_str.split())} tokens to " + str(len(ref_str.split('\n'))) + " references")

        # Perform alignment
        try:
            result = align_texts(ref_str, hyp_str, is_tokenized=is_tokenized)
            
            # Output result
            resulting_translations = {}
            for input_ref, line in zip(  refs, result.split("\n")):
                if segmenter is not None and not no_detok:
                    line = segmenter.decode(line)
                #print(line)
                resulting_translations[input_ref] = line
            
            return resulting_translations
                
        except Exception as e:
            logger.fatal(f"Error: {e}")
            return None

@dataclass
class MergedData:
    """Schema for the combined data used in evaluation."""
    dataset_id: str
    sample_id: int
    src_lang: str
    tgt_lang: str
    output: str
    src_ref: Optional[str]
    tgt_ref: Optional[Dict[str, Any]]
    doc_id: str
    src_audio: Optional[str]
    benchmark_metadata: Optional[Dict[str, Any]]


def main():
    """
    Main function for model evaluation process.
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Setup robust argument parsing
    parser = argparse.ArgumentParser(description='Run mweralign.')
    parser.add_argument('--manifest-long-path', type=Path, required=True, help='Path to the manifest file (input data).')
    parser.add_argument('--manifest-short-path', type=Path, required=True, help='Path to the manifest file (input data).')
    parser.add_argument('--output-path', type=Path, required=True, help='Path to the file with model predictions.')
    parser.add_argument('--output-segmented-file', type=Path, required=True, help='Output file for segmented outputs (JSONL).')
    
    args = parser.parse_args()

    # load jsonl files
    long_manifest = load_jsonl(args.manifest_long_path)
    short_manifest = load_jsonl(args.manifest_short_path)
    output = load_jsonl(args.output_path)

    # first align output and manifest-long
    merged_data = []
    for i, (input_item, output_item) in enumerate(zip(long_manifest, output)):

        merged_entry = MergedData(
            dataset_id=output_item['dataset_id'],
            sample_id=output_item['sample_id'],
            src_lang=output_item['src_lang'],
            tgt_lang=output_item['tgt_lang'],
            output=output_item['output'],
            src_ref=input_item.get('src_ref'),
            tgt_ref=input_item.get('tgt_ref'),
            src_audio=input_item.get('src_audio'),
            doc_id=input_item.get('doc_id'),
            benchmark_metadata=input_item.get('benchmark_metadata')
        )
        merged_data.append(merged_entry)

    # get source and references segmented
    for item_merged_data in merged_data:
        dict_segmented = {item['src_ref']: item['tgt_ref'] for item in short_manifest if item['doc_id'] == item_merged_data.doc_id }
        item_merged_data.references_segmented = dict_segmented

    # align using mweralign!
    all_alignments = {}
    for item_merged_data in merged_data:

        refs = list(item_merged_data.references_segmented.values())
        hyp  = [item_merged_data.output]

        alignments = segment(refs, hyp, item_merged_data.tgt_lang, no_detok=False)
        all_alignments.update(alignments)

    OUTPUT = []
    for item_short in short_manifest:
        output_entry = {
            "dataset_id": item_short['dataset_id'], 
            "sample_id": item_short['sample_id'], 
            "src_lang": item_short['src_lang'],
            "tgt_lang": item_short['tgt_lang'], 
            "output": all_alignments[ item_short['tgt_ref'] ]
        }

        OUTPUT.append(output_entry)
    
    with open(args.output_segmented_file, "w", encoding="utf-8") as f:
        for entry in OUTPUT:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()