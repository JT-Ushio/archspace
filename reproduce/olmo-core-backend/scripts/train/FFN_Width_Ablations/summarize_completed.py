"""Summarize only the final 1000 training steps, without checkpoint evaluation."""

import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "analysis/completed_20261009"
ORDER = ["C2", "C6", "W1", "W1QK", "W1M", "W2", "G1"]
DISPLAY = {"W1QK": "W1-QK", "W1M": "W1-M"}
SHAPES = {
    "C2": (8192, 568, 520, "Linear / SiLU"),
    "C6": (8192, 568, 520, "SiLU / SiLU"),
    "W1": (6144, 736, 640, "Linear / SiLU"),
    "W1QK": (6144, 736, 640, "SiLU / SiLU"),
    "W1M": (6144, 568, 520, "Linear / SiLU"),
    "W2": (4096, 976, 944, "Linear / SiLU"),
    "G1": (8192, 568, 520, "可学习，初始 0 / 1"),
}


def main():
    stats = json.loads((DATA / "FULL_TRAIN_CE.json").read_text())
    assert stats["complete"] and stats["end_step"] == 14306
    assert stats["tail_start"] == 13307 and stats["clean_tail_n"] == 1000
    with (DATA / "train_ce_by_step.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in history] == list(range(1, 14307))
    assert not any(row["common_excluded"] == "True" for row in history[-1000:])
    rows = stats["runs"]
    for name in ORDER:
        tail = [float(row[name]) for row in history[-1000:]]
        assert all(math.isfinite(value) and value > 0 for value in tail)
        assert math.isclose(sum(tail) / 1000, rows[name]["common_clean_last1000_mean"], abs_tol=1e-12)
    loss = {name: rows[name]["common_clean_last1000_mean"] for name in ORDER}
    summary = {
        "scope": "Final 1000-step Train CE only; no checkpoint evaluation required",
        "window": [13307, 14306], "points_per_run": 1000,
        "ranking": sorted(ORDER, key=loss.get),
        "runs": {name: {"parameters": rows[name]["parameters"], "mean_train_ce": loss[name],
                        "delta_vs_c2": loss[name] - loss["C2"],
                        "delta_vs_c6": loss[name] - loss["C6"]} for name in ORDER},
        "width_recovery_w1_minus_w1m": loss["W1"] - loss["W1M"],
        "q_silu_effect_original": loss["C6"] - loss["C2"],
        "q_silu_effect_w1": loss["W1QK"] - loss["W1"],
        "q_activation_interaction": stats["q_activation_interaction"],
    }
    (HERE / "TAIL_TRAIN_CE_COMPARISON.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 五个新实验：末期Train CE分析", "", "分析日期：2026-10-09。按用户最新要求，仅分析训练CE，不等待或继续checkpoint评测。", "",
             "**缩宽并回补rank的W1/W2没有超过C2/C6；G1的末1000步Train CE最低，比C6低0.001031。后者只有一次新运行，暂作小幅改善候选。**", "",
             "## 配置与结果", "",
             "所有配置UG完全共享输入因子、中间Linear，保留原生SwiGLU；Down均无gain RMSNorm。Attention独立，Q/K/V rank512、O rank496，V/O中间Linear。", "",
             "| 配置 | FFN宽度 | UG共享rank | Down rank | Q/K中间变换 | 参数量 | 末1000步Train CE | 相对C6（负为更好） |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |"]
    for name in ORDER:
        m, ug, down, qk = SHAPES[name]
        lines.append(f"| {DISPLAY.get(name, name)} | {m} | {ug} | {down} | {qk} | {rows[name]['parameters']:,} | {loss[name]:.6f} | {loss[name]-loss['C6']:+.6f} |")
    lines += ["", "排序：" + " < ".join(DISPLAY.get(name, name) for name in summary["ranking"]) + "。", "",
              "W1-M少54,263,808参数（约6.808%），是解释性小模型对照，不参与严格同预算选型。G1多32个标量，近似同预算，不称逐个参数完全匹配。", "",
              "## 1. 缩宽的损失，瓶颈回补只能部分弥补", "",
              f"C2→W1-M：只将展开宽度8192降为6144，UG568/Down520不变，CE增加{loss['W1M']-loss['C2']:.6f}。",
              f"W1-M→W1：展开宽度仍为6144，UG568→736、Down520→640，CE降低{loss['W1M']-loss['W1']:.6f}。回补容量有效，但W1仍比C2高{loss['W1']-loss['C2']:.6f}、比C6高{loss['W1']-loss['C6']:.6f}。",
              f"W2进一步缩至4096、UG976/Down944，虽然严格保持797,050,880参数，CE仍比C2高{loss['W2']-loss['C2']:.6f}。更大的瓶颈没有补偿更窄的展开空间。", "",
              "本轮不支持缩宽方案带来质量收益。已经测试的组合中继续保留8192/共享568/Down520；结果不能证明8192是全局最优，也不能将组合劣化全部归因于某一个维度。Jacobian秩上界增大不保证实际质量改善。", "",
              "## 2. Q SiLU没有救回W1", "",
              f"原背景C6−C2={loss['C6']-loss['C2']:+.6f}；缩宽背景W1-QK−W1={loss['W1QK']-loss['W1']:+.6f}。后者几乎持平、略差，没有恢复原背景的微小收益。交互差分={stats['q_activation_interaction']:+.6f}，只是单次运行的描述，不称统计显著。", "",
              "W1/W1-QK同shape、同硬件分配，实际GPU step0全部275张参数字节SHA256一致；因此两条是本轮干净的Q激活单变量对照。", "",
              "## 3. G1是值得保留的候选，收益仍小", "",
              r"G1保持8192/共享568/Down520，学习每层Q/K的 $\phi_\alpha(a)=a+\alpha(\mathrm{SiLU}(a)-a)$；Q从0、K从1开始，共新增32个参数。", "",
              f"末期CE为{loss['G1']:.6f}，比C2低{loss['C2']-loss['G1']:.6f}、比C6低{loss['C6']-loss['G1']:.6f}。这是本轮最好的末期训练结果，但改善幅度约0.001，每配置只有一次新训练；建议作为重复实验候选，而非直接认定稳定新最优。", "",
              "## 比较口径与边界", "",
              "- 五条均完成30B/14306步；使用完整原始W&B记录中的steps13307–14306，各1000点，无缺失、重复或非有限CE。",
              "- C2/C6使用已冻结、校验SHA256的完整参考历史。C6异常step12516在末期窗口外，不影响本次排名。",
              "- 比较未平滑的末1000步平均值；单个最终step的batch波动不作为选型依据。",
              "- 序列8192、全局batch256、Muon LR0.005、WSD warmup2000/末20% decay、FP8及数据seed一致；原始C2/C6为16卡，新五条为pool8卡H200。",
              "- 不同shape即使运行seed相同，也不保证其他参数初始字节一致。除W1/W1-QK外，保留初始化与硬件差异，不能作纯单变量因果解释。",
              "- 本文只对Train CE选型，不据此推导下游benchmark、泛化、显存或吞吐收益。本次没有提交新的训练。", "",
              "## W&B完整名称", "", "| 配置 | W&B实验名称 |", "| --- | --- |"]
    for name in ORDER:
        run = stats["provenance"][name]["run_name"]
        lines.append(f"| {DISPLAY.get(name, name)} | [{run}](https://wandb.ai/archspace/MoSE/runs/{run}) |")
    lines += ["", "[原始完整CE与来源哈希](analysis/completed_20261009/FULL_TRAIN_CE.json) / [逐step原始CSV](analysis/completed_20261009/train_ce_by_step.csv) / [末期比较JSON](TAIL_TRAIN_CE_COMPARISON.json) / [详细设计](EXPERIMENT_DESIGN.md) / [运行记录](RUN_STATUS.md)。", ""]
    (HERE / "COMPLETED_RESULTS_20261009.md").write_text("\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
