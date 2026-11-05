# -*- coding: utf-8 -*-
"""
cam_test.py —— 摄像头实时手语测试（方案B 链路，本地显示，不录像不外传）

操作：
  1) python cam_test.py        # 弹出摄像头窗口（按 q 随时退出）
  2) 正对摄像头，逐个字母打手语，每个字母保持约 1 秒
  3) 一个词拼完后放下手停顿约 0.5 秒，再拼下一个词
  推荐练习词：nali(4字母) / yibao(5字母) / xiexie(6字母) / banli(6字母)

窗口信息：
  frame   —— 当前帧识别到的字母与置信度
  letters —— 已提交的字母流（连续 2 帧以上相同才提交）
  words   —— 按 vocab.json 拼读出的拼音整词
  底部蓝字 —— pinyin_codec 词表还原的中文

注意：同一字母需要连打两次时（如 aa），请把手放下停顿后再打第二次。
"""
import sys
import time

import cv2
import numpy as np

import sign_real_link as srl
import pinyin_codec as pc

T_LIMIT = 180  # 秒，防止忘记关窗口


def put(img, text, org, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


try:
    from PIL import Image as PILImage, ImageDraw, ImageFont
    FONT_CN = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 24)
except Exception:
    FONT_CN = None


def cn_overlay(img, line):
    if FONT_CN is None or not line:
        return img
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    d.rectangle([0, img.shape[0] - 38, img.shape[1], img.shape[0]], fill=(8, 14, 28))
    d.text((8, img.shape[0] - 33), line, font=FONT_CN, fill=(110, 215, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[X] 打不开摄像头：请先关闭占用相机的程序（微信视频、相机应用）再重试。")
        sys.exit(2)

    targets = srl.load_pinyin_targets()
    committed, cconfs = [], []
    last_letter, run, run_conf = None, 0, 0.0
    cur_letter, cur_conf = None, 0.0
    t0 = time.time()
    warn_once = False
    print("[cam_test] 窗口已打开：逐个字母打手语（每个保持约1秒），按 q 结束 ...")

    while time.time() - t0 < T_LIMIT:
        ok, frame = cap.read()
        if not ok:
            print("[X] 摄像头读取失败（可能被其他程序抢占）")
            break
        small = cv2.resize(frame, (640, 480))
        box = None
        try:
            box = srl.best_box_of_frame(small)
            if box is not None:
                cur_letter, cur_conf = srl.classify_crop(small, box)
            else:
                cur_letter, cur_conf = None, 0.0
        except Exception as e:
            if not warn_once:
                print("[warn] 推理异常:", e)
                warn_once = True
            cur_letter, cur_conf = None, 0.0

        # 连续 run 提交字母
        if cur_letter and cur_letter == last_letter:
            run += 1
            run_conf = max(run_conf, cur_conf)
        elif cur_letter:
            last_letter, run, run_conf = cur_letter, 1, cur_conf
        else:
            last_letter, run, run_conf = None, 0, 0.0

        if cur_letter and run >= 2 and (not committed or committed[-1] != cur_letter):
            committed.append(cur_letter)
            cconfs.append(round(run_conf, 3))
            run = 1  # 提交后保持低 run，允许停顿后再次提交同字母

        words = srl.spell_letters(committed, targets)
        try:
            hanzi = " ".join(pc.decode_tokens(words)["words"])
        except Exception:
            hanzi = ""

        # 绘制
        if box:
            cv2.rectangle(small, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 255, 0), 2)
        put(small, f"frame: {cur_letter or '-'}  conf: {cur_conf:.2f}", (8, 24), 0.65, (0, 255, 255))
        put(small, "letters: " + "-".join(committed[-14:]), (8, 50), 0.6)
        put(small, "words: " + " ".join(words), (8, 76), 0.6, (140, 255, 140))
        put(small, f"{int(time.time() - t0)}s/{T_LIMIT}s | q=quit | hold each letter ~1s",
            (8, 470), 0.5, (200, 200, 200))
        small = cn_overlay(small, ("中文: " + hanzi) if hanzi else "")
        cv2.imshow("Sign Cam Test (q to quit)", small)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    words = srl.spell_letters(committed, targets)
    d = pc.decode_tokens(words)
    print("\n===== 总结 =====")
    print("用时:", round(time.time() - t0, 1), "s")
    print("提交字母流:", "-".join(committed) or "(无，手未保持够久或未检测到手)")
    print("字母置信度:", cconfs)
    print("拼读拼音词:", words)
    print("词表中文:", d["words"])
    print("词表外丢弃:", d["unknown"], "| 模糊命中:", d["fuzzy"])
    print("确认信号:", d["confirmed"])


if __name__ == "__main__":
    main()
