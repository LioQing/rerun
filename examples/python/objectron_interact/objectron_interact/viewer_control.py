"""Route-B raw gRPC client for the Rerun viewer's `SetTimeCursor` method.

Uses `grpclib` against pre-generated protobuf stubs in the sibling
`viewer_proto` subpackage — no SDK rebuild needed.
"""

from __future__ import annotations

from grpclib.client import Channel

from .viewer_proto.v1alpha1 import common_pb2, viewer_grpc, viewer_pb2

DEFAULT_VIEWER_PORT: int = 9876


async def seek_viewer(
    *,
    timeline: str,
    time: int,
    viewer_port: int = DEFAULT_VIEWER_PORT,
    play: bool = False,
) -> None:
    """Move the viewer's time cursor to `time` on `timeline`.

    Parameters
    ----------
    timeline:
        Timeline name (e.g. ``"time"`` for the duration timeline or ``"frame"``
        for the sequence timeline logged by the objectron example).
    time:
        Time value: a sequence index for ``sequence`` timelines, or nanoseconds
        for ``duration``/``timestamp`` timelines.
    viewer_port:
        gRPC port of the viewer's ``ViewerControlService`` (default 9876).
    play:
        Start playback from the new position instead of pausing.
    """
    channel = Channel(host="127.0.0.1", port=viewer_port)
    try:
        stub = viewer_grpc.ViewerControlServiceStub(channel)
        request = viewer_pb2.SetTimeCursorRequest(
            timeline=common_pb2.Timeline(name=timeline),
            time=common_pb2.TimelineTime(time=time),
            play=play,
        )
        await stub.SetTimeCursor(request)
    finally:
        channel.close()