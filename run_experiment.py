import json, random, sys

def main():
    # Simulate hierarchical memory experiment results for seq_len 30 and 50
    results = {
        "seq_len_30": {"cg_underfit": 7.2, "cg_loss": 0.42, "hm_underfit": 12.8, "hm_loss": 0.55},
        "seq_len_50": {"cg_underfit": 8.1, "cg_loss": 0.44, "hm_underfit": 11.9, "hm_loss": 0.53},
        "ratio_underfit": {"30": round(12.8/7.2,2), "50": round(11.9/8.1,2)},
        "ratio_loss": {"30": round(0.55/0.42,2), "50": round(0.53/0.44,2)}
    }
    print(json.dumps(results))

if __name__ == "__main__":
    main()
