#!/usr/bin/env python3
"""Send one text message to the device agent and print its reply.

This is the text half of the agent's WebSocket surface. The voice bridge
drives the same socket from audio; this drives it from the command line, so
the agent can be exercised without the microphone, the ASR or the PPP link
being involved at all.

    python3 ws_chat.py "请巡检设备"
    python3 ws_chat.py "现在几点了" --wait 30

The agent answers in parts: an acknowledgement first, then the result once the
tool loop finishes. There is no marker saying which part is the last one, so
this keeps reading until the socket has been quiet for QUIET_S and prints
every part as it arrives.

Exits 0 once a reply arrives, 1 if none did.
"""
import asyncio
import json
import sys
import time

import websockets

URL = "ws://192.168.223.2:28789"
CHAT_ID = "ws-chat"
DEFAULT_WAIT = 120.0
QUIET_S = 15.0         # stop once the socket has been silent this long
                       # A real LLM answer arrives after several tool round
                       # trips with gaps between them; 4 s cut the reply off
                       # mid-turn and made a working endpoint look broken.


async def main(text: str, wait: float) -> int:
    t0 = time.time()
    got = False
    try:
        # ping_interval=None for the same reason the bridge sets it: the
        # device serialises audio, replies and pongs behind one lock, so a
        # pong can arrive late and the library would drop a healthy session.
        async with websockets.connect(URL, max_size=None,
                                      ping_interval=None) as ws:
            await ws.send(json.dumps(
                {"type": "message", "content": text, "chat_id": CHAT_ID}))
            print("[ws_chat] sent: %s" % text)

            # A tool-using turn can take tens of seconds, so the first part
            # gets a long wait; after that, silence is what says it is over.
            budget = wait
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=budget)
                except asyncio.TimeoutError:
                    break
                budget = QUIET_S
                if isinstance(raw, bytes):
                    continue          # audio frames belong to the voice bridge
                data = json.loads(raw)
                if data.get("type") == "response":
                    print("[dev] %s" % data.get("content", ""))
                    got = True
    except OSError as exc:
        print("[ws_chat] cannot reach the agent at %s: %s" % (URL, exc))
        return 1
    except websockets.exceptions.WebSocketException as exc:
        print("[ws_chat] websocket error: %s" % exc)
        return 1

    if not got:
        print("[ws_chat] no response within %.0fs" % wait)
        return 1
    print("[ws_chat] (%.0fs)" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    wait = DEFAULT_WAIT
    if "--wait" in argv:
        i = argv.index("--wait")
        try:
            wait = float(argv[i + 1])
        except (IndexError, ValueError):
            print("--wait needs a number of seconds")
            sys.exit(2)
        del argv[i:i + 2]

    if not argv:
        print(__doc__)
        sys.exit(2)

    sys.exit(asyncio.run(main(" ".join(argv), wait)))
