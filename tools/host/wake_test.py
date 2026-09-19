#!/usr/bin/env python3
"""Check the wake-phrase matcher against what the recogniser actually returns.

Runs entirely offline -- no device, no audio. The positive cases are the real
transcripts captured while testing with the board, which is the point: they
are what "你好openvela" comes back as over a 5801 Hz link, and the matcher has
to accept all of them. Do not "tighten" this by putting the brand back in
without re-reading voice_bridge.WAKE_ANCHOR first.

    python3 wake_test.py
"""
import sys

from voice_bridge import split_wake, understand

# Transcripts the recogniser really produced for "你好openvela", plus the
# clean spellings that would appear if the link ever carried them.
POSITIVE = [
    "你好 open renline",
    "你好 open rula",
    "你好 ok v 乐",
    "你好 oppo via",
    "你好，openvela",
    "你好openvela，现在几点了",
]

# Things that must not wake the device. The first two are the ones that make
# the rule as tight as it is: "你好" alone is an existing command, and the
# rest is the room noise the recogniser turns into fluent-looking Chinese.
NEGATIVE = [
    "你好",
    "你好，介绍一下你自己",
    "然后我觉得你",
    "中年卫地",
    "我把堆了好多码了",
    "哎呦我的天",
    "好被告人是我和审判员能够为庭法庭庭庭",
    "openvela",
    "挂",
    "内存",
]


def main() -> int:
    fails = 0

    print("=== must wake ===")
    for text in POSITIVE:
        heard, tail = split_wake(text)
        print("  %-28s %s  tail=%r" % (text, "ok " if heard else "FAIL", tail))
        fails += 0 if heard else 1

    print("\n=== must not wake ===")
    for text in NEGATIVE:
        heard, _ = split_wake(text)
        print("  %-28s %s" % (text, "ok " if not heard else "FAIL (false wake)"))
        fails += 0 if not heard else 1

    # A wake phrase with a command in the same breath has to survive as a
    # command, or the speaker has to say everything twice.
    print("\n=== same-breath command still routes ===")
    _, tail = split_wake("你好openvela，现在几点了")
    cmd = understand(tail)
    print("  tail=%r -> %r" % (tail, cmd))
    fails += 0 if cmd else 1

    # "你好" on its own must keep working as the command it already was.
    print("\n=== bare 你好 still means introduce-yourself ===")
    cmd = understand("你好")
    print("  -> %r" % cmd)
    fails += 0 if cmd else 1

    print("\nfailures: %d" % fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
