import json


# reference: https://github.com/informagi/REL/blob/921174b0c3c6526273eefa51385312e6a3e190eb/REL/utils.py#L67
def process_results(mentions_dataset, predictions, processed, include_offset=False,):
    """
    Function that can be used to process the End-to-End results.
    :return: dictionary with results and document as key.
    """
    res = {}
    for doc in mentions_dataset:
        if doc not in predictions:
            # No mentions found, we return empty list.
            continue
        pred_doc = predictions[doc]
        ment_doc = mentions_dataset[doc]
        text = processed[doc][0]
        res_doc = []

        for pred, ment in zip(pred_doc, ment_doc):
            sent = ment["sentence"]
            idx = ment["sent_idx"]
            start_pos = ment["pos"]
            mention_length = int(ment["end_pos"] - ment["pos"])

            if pred["prediction"] != "NIL":
                temp = (
                    start_pos,
                    mention_length,
                    ment["ngram"],
                    pred["prediction"],
                    pred["conf_ed"],
                    ment["conf_md"] if "conf_md" in ment else 0.0, # filter by this value (>= 0.9)
                    ment["tag"] if "tag" in ment else "NULL",
                )
                res_doc.append(temp)
        res[doc] = res_doc
    return res


# reference: https://github.com/informagi/REL/blob/f6d20e7388a09ce4300f7606668568fd1c5e5926/scripts/code_tutorials/predict_EL.py#L7
def dataset_preprocessing(list_examples, e=0):
    """
    Function that can be used to prepare the input text.
    :return: dictionary with the input text.
    """
    spans = []
    return { i+e: [example.rstrip(), spans] for i, example in enumerate(list_examples) }


def export_results(fname, res):
    """
    Function that can be used to export the preprocessed results
    :input: filename and dictionary with the results stored in a dictionary
    """
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=4)


def import_results(fname):
    """
    Function that can be used to import the preprocessed results from a json file
    :return: results
    """
    with open(fname, 'r', encoding='utf-8') as f:
        res = json.load(f)
        return res
