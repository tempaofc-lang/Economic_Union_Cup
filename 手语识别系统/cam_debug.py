# -*- coding: utf-8 -*-
"""
cam_debug.py —— 摄像头诊断工具（查"为什么老识别成 U"）

用法：
    python cam_debug.py

它会：
  1. 连续 20 秒从摄像头抓帧；
  2. 每 0.4 秒打印一次该帧「最可能的 3 个字母 + 置信度」；
  3. 统计这 20 秒里每个字母出现了多少次、平均置信度多少。

重点看：
  - 你比 B 的时候，U 排第几？如果 U 是第 2 名且分数接近，说明是"形近误判"；
  - 你手放下的时候，还在输出 U 吗？如果还在，说明模型把手以外的东西当成了手。

不需要 .env，不调用任何云端接口，纯本地。
"""
import time
import cv2
import numpy as np

import sign_real_link as s


def top3_of(yolo, frame, box):
    """取该框位置的 top-3 候选类别。YOLO 的 boxes 只给最优类，
    这里直接对整帧跑一次拿所有候选。"""
    res = yolo(frame, imgsz=640, conf=0.15, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    out = []
    for b in res.boxes:
        out.append((str(yolo.names[int(b.cls)]).upper(), float(b.conf)))
    out.sort(key=lambda x: -x[1])
    return out[:3]


def main():
    print("=" * 62)
    print(" 摄像头识别诊断（20 秒）")
    print(" 请依次做：① 手放下 5 秒  ② 比 B 5 秒  ③ 比 L 5 秒  ④ 比 U 5 秒")
    print("=" * 62)

    yolo = s.get_yolo_model()
    cap, desc = s.open_camera()
    print(f"摄像头：{desc}\n")

    stats = {}
    detail = []
    t0 = time.time()
    last_print = 0.0
    try:
        while time.time() - t0 < 20:
            ok, frame = cap.read()
            if not ok:
                continue
            now = time.time() - t0
            if now - last_print < 0.4:
                continue
            last_print = now

            res = yolo(frame, imgsz=640, conf=0.15, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                line = "手放下/未检出"
                print(f"[{now:5.1f}s] {line}")
                continue

            b = max(res.boxes, key=lambda x: float(x.conf))
            letter = str(yolo.names[int(b.cls)]).upper()
            conf = float(b.conf)

            # 同一帧把所有框的候选都收集（有些帧会给出多个类）
            cands = [(str(yolo.names[int(x.cls)]).upper(), float(x.conf)) for x in res.boxes]
            cands.sort(key=lambda x: -x[1])
            txt = "  ".join(f"{n}:{c:.2f}" for n, c in cands[:3])
            print(f"[{now:5.1f}s] 主判 {letter}({conf:.2f})   候选: {txt}")

            stats.setdefault(letter, []).append(conf)
            detail.append((round(now, 1), letter, round(conf, 3),
                           [(n, round(c, 3)) for n, c in cands[:3]]))
    finally:
        cap.release()

    print("\n" + "=" * 62)
    print(" 统计结果")
    print("=" * 62)
    if not stats:
        print("整个 20 秒都没有检出字母。")
        return

    total = sum(len(v) for v in stats.values())
    for k, v in sorted(stats.items(), key=lambda x: -len(x[1])):
        pct = len(v) / total * 100
        print(f"  {k:3s}  出现 {len(v):3d} 次 ({pct:5.1f}%)   平均置信度 {np.mean(v):.2f}")

    # 找 U 与其它字母的共存情况
    print("\n" + "-" * 62)
    u_with_others = [d for d in detail if d[1] == "U" and len(d[3]) > 1]
    if u_with_others:
        print(f"判定为 U 的帧中，有 {len(u_with_others)} 帧同时出现了其它候选：")
        for t, letter, conf, cands in u_with_others[:8]:
            print(f"  [{t}s] U:{conf:.2f}  其它候选 {cands[1:]}")
    else:
        print("判定为 U 的帧里，U 都是唯一的候选（说明模型非常确信，是手势本身像 U）")

    print("\n如果 'U' 占比超过 40%，且你的手明明是别的姿势，")
    print("请把上面完整的输出截图发给开发者。")


if __name__ == "__main__":
    main()
