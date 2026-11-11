# -*- coding: utf-8 -*-
"""
cam_calib.py —— 开跑前 5 秒自检，直接告诉你「现在这个位置能不能跑」。

为什么需要它：
  YOLO26 的字母置信度对「手在画面里占多大」极其敏感。实测（40 张验证图，
  详见 改动说明.md 附录）：
      手占画面 100% → conf 0.65    手占画面 40% → conf 0.63（检出率掉）
      手占画面 30% → conf 0.51     手占画面 25% → conf 0.05，基本全漏
  换句话说：把 MIN_RUN 从 2 调到 3 不会让 0.4 变成 0.7，只会让本来就
  识别不出的帧被更严格地丢掉 —— 屏幕上会变成「什么都不出」。

  真正决定成败的是三件事，这个脚本逐条检查：
      1) 手在画面里够不够大（目标：手框长边 ≥ 画面短边的 55%）
      2) 画面够不够清楚（边缘清晰度，正常应 ≥ 60，对焦糊/逆光会掉到 20 以下）
      3) 光照够不够（平均亮度 90~190 最好，太暗太亮都掉点）

用法：
    python cam_calib.py                 # 用 0 号摄像头
    python cam_calib.py --cam 1         # 第二台摄像头
    python cam_calib.py --seconds 6     # 采集时长

判读：
    [OK]   可以直接录演示视频
    [调整] 按提示调整后重跑本脚本（直到三项都是 OK）
    [换机] 手已经贴到最近还是不够大 → 换外接摄像头 / 换一台笔记本
"""
import argparse
import time

import cv2
import numpy as np

import sign_real_link as srl


def sharpness(frame):
    """拉普拉斯方差：越大越清晰。1080p 正常 100+，糊了会掉到 20 以下。"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--min-ratio", type=float, default=0.55,
                    help="手框长边 / 画面短边 的最低要求")
    args = ap.parse_args()

    # 让模型先加载，避免把加载耗时的帧当成测速
    srl.get_yolo_model()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[X] 打不开摄像头 %d：先关掉微信/钉钉/腾讯会议/相机应用再试。" % args.cam)
        raise SystemExit(2)
    # 有些摄像头/驱动不支持 set，失败不影响检测，只是不强制分辨率
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass

    print("=" * 62)
    print("开跑前自检：请把手举到画面中央，摆一个字母（比如 B），保持不动")
    print("采集 %g 秒 ..." % args.seconds)
    print("=" * 62)

    ratios, confs, sharps, brights, fps_cnt = [], [], [], [], 0
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, 480))
        fps_cnt += 1
        box = srl.best_box_of_frame(frame)

        sh = sharpness(frame)
        br = float(frame.mean())
        sharps.append(sh)
        brights.append(br)

        if box is not None:
            side = max(box["x2"] - box["x1"], box["y2"] - box["y1"])
            ratio = side / float(min(frame.shape[:2]))
            ratios.append(ratio)
            confs.append(float(box["conf"]))
            cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]),
                          (0, 255, 0), 2)
            cv2.putText(frame, "%s %.2f | ratio %.2f" % (box["cls_name"], box["conf"], ratio),
                        (box["x1"], max(18, box["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no hand detected - move closer", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if time.time() - last_print > 0.3:
            last_print = time.time()
            txt = "sharp %.0f | bright %.0f | hand %s" % (
                sh, br, ("%.2f" % ratios[-1]) if ratios else "-")
            cv2.putText(frame, txt, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 0), 2)
        cv2.imshow("Calibration (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    elapsed = time.time() - t0
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 62)
    print("自检结果")
    print("=" * 62)
    if not ratios:
        print("[X] 整段没检测到手。")
        print("    → 把手举到镜头正前方，与镜头距离约 40~60cm，手掌正对镜头。")
        print("    → 检查窗口里是否显示画面（黑屏说明摄像头被占用或选错 --cam）。")
        return

    r_med = float(np.median(ratios))
    c_med = float(np.median(confs))
    s_med = float(np.median(sharps))
    b_med = float(np.median(brights))
    det_rate = len(ratios) / max(1, fps_cnt) * 100
    fps = fps_cnt / elapsed if elapsed > 0 else 0

    ok_all = True

    # 1) 手的大小
    if r_med >= args.min_ratio:
        print("[OK]   手占画面 %.2f（≥%.2f）—— 大小合适" % (r_med, args.min_ratio))
    elif r_med >= 0.40:
        print("[调整] 手占画面仅 %.2f，偏低。往镜头靠近约 10cm，再跑一次。" % r_med)
        ok_all = False
    elif r_med >= 0.30:
        print("[调整] 手占画面只有 %.2f，太远了。往前坐/站起来，手掌占满画面中间一半。" % r_med)
        ok_all = False
    else:
        print("[换机] 手占画面只有 %.2f，已经贴到最近还是这么小 —— 这台笔记本内置摄像头" % r_med)
        print("       视野太广，这个模型用不了。换外接摄像头（或换一台机器）。")
        ok_all = False

    # 2) 清晰度（阈值按 640x480 实测定：清晰 450+，轻微虚焦 75 左右，
    #    明显虚焦 16 左右；同一批图置信度随之下滑 0.65 → 0.55 → 0.44）
    if s_med >= 150:
        print("[OK]   清晰度 %.0f（≥150）" % s_med)
    elif s_med >= 40:
        print("[调整] 清晰度只有 %.0f，画面有点虚：擦干净镜头、靠近一点、" % s_med)
        print("       关掉背光（别对着窗户），再跑一次。")
        ok_all = False
    else:
        print("[调整] 清晰度只有 %.0f，严重虚焦/镜头太脏/光线太暗。" % s_med)
        ok_all = False

    # 3) 亮度
    if 90 <= b_med <= 195:
        print("[OK]   亮度 %.0f（90~195）" % b_med)
    elif b_med < 90:
        print("[调整] 亮度 %.0f 太暗：面向光源（窗户/台灯），别背光。" % b_med)
        ok_all = False
    else:
        print("[调整] 亮度 %.0f 过曝：避开正对强光/窗口天光。" % b_med)
        ok_all = False

    print("-" * 62)
    print("当前字母置信度中位数 %.2f（低于 0.50 时，单帧根本过不了阈值）" % c_med)
    print("检出率 %.0f%%   采集帧率 %.1f FPS" % (det_rate, fps))
    if c_med < 0.50:
        print("  → 置信度偏低。先解决上面标 [调整]/[换机] 的项，再跑本脚本复查。")
        print("    注意：调 MIN_RUN（去抖帧数）不会提高置信度，只会让丢帧更多。")
    print("-" * 62)
    if ok_all and c_med >= 0.50:
        print("结论：这台机器 + 这个位置可以录演示视频 → python cam_test.py")
    else:
        print("结论：先按上面的提示调整，再跑一次 python cam_calib.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
