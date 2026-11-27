import json
from jiwer import cer
from predict_inscription import InscriptionAnalyzer

TEST_FILE = "test.jsonl"


def evaluate(max_samples: int | None = 15):
    analyzer = InscriptionAnalyzer(train_only_index=True)
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f]

    if max_samples:
        test_samples = test_samples[:max_samples]

    cer_scores = []
    period_hits, lang_hits, script_hits = 0, 0, 0
    total = len(test_samples)

    for idx, sample in enumerate(test_samples, start=1):
        truth = json.loads(sample["target_json"])
        out = analyzer.predict(sample["image_path"], stream=False)
        pred = out["prediction"]

        gt_trans = truth.get("transliteration", "")
        pr_trans = pred.get("transliteration", "")
        sample_cer = cer(gt_trans, pr_trans) if gt_trans else None
        if sample_cer is not None:
            cer_scores.append(sample_cer)

        if pred.get("period", "").strip().lower() == truth.get("period", "").strip().lower():
            period_hits += 1
        if pred.get("language", "").strip().lower() == truth.get("language", "").strip().lower():
            lang_hits += 1
        if pred.get("script", "").strip().lower() == truth.get("script", "").strip().lower():
            script_hits += 1

        cer_str = f"{sample_cer:.3f}" if sample_cer is not None else "N/A"
        print(f"[{idx}/{total}] #{sample['inscriptionNo']} | CER: {cer_str} | Pred Script: {pred.get('script', 'N/A')}")

    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0.0
    print("\n=== Final Held-Out Real Test Set Results ===")
    print(f"Samples Evaluated: {total}")
    print(f"Transliteration Character Error Rate (CER): {avg_cer:.2%}")
    print(f"Period Accuracy:   {period_hits / total:.2%}")
    print(f"Language Accuracy: {lang_hits / total:.2%}")
    print(f"Script Accuracy:   {script_hits / total:.2%}")


if __name__ == "__main__":
    # Runs 15 test samples (~2-3 mins); change to max_samples=None for all 107
    evaluate(max_samples=15)