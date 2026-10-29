"""List video capture devices in Media Foundation order, which is the order OpenCV's CAP_MSMF indices use.

DirectShow (and so pygrabber) can list the same cameras in a different order.
"""

import ctypes
import uuid
from ctypes import POINTER, byref, c_uint32, c_void_p, c_wchar_p


class GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16), ("Data3", ctypes.c_uint16),
                ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "GUID":
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE = GUID.parse("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID = GUID.parse("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME = GUID.parse("60d0e559-52f8-4fa2-bbce-acdb34a8ec01")
MF_VERSION = 0x00020070
COINIT_APARTMENTTHREADED = 0x2

# Vtable slots: IUnknown is 0-2, then IMFAttributes in declaration order (IMFActivate inherits it).
RELEASE = 2
GET_ALLOCATED_STRING = 13
SET_GUID = 24

ole32 = ctypes.OleDLL("ole32")
mfplat = ctypes.OleDLL("mfplat")
mf = ctypes.OleDLL("mf")
co_task_mem_free = ctypes.windll.ole32.CoTaskMemFree
co_task_mem_free.argtypes = [c_void_p]
co_task_mem_free.restype = None


def com_call(obj: c_void_p, slot: int, argtypes: tuple, *args):
    """Call the COM method in vtable slot on obj. HRESULT failures raise OSError."""
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    method = ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtable[slot])
    return method(obj, *args)


def release(obj: c_void_p) -> None:
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    ctypes.WINFUNCTYPE(ctypes.c_ulong, c_void_p)(vtable[RELEASE])(obj)


def list_msmf_cameras() -> list[str]:
    """Camera names, where a name's position is its cv2.CAP_MSMF index."""
    try:
        ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED)
    except OSError:
        pass  # Already initialized on this thread in another mode, which works just as well here.
    mfplat.MFStartup(MF_VERSION, 0)
    names = []
    attributes = c_void_p()
    try:
        mfplat.MFCreateAttributes(byref(attributes), 1)
        com_call(attributes, SET_GUID, (POINTER(GUID), POINTER(GUID)),
                 byref(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE), byref(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))
        devices = POINTER(c_void_p)()
        count = c_uint32()
        mf.MFEnumDeviceSources(attributes, byref(devices), byref(count))
        for i in range(count.value):
            device = c_void_p(devices[i])
            name = c_wchar_p()
            length = c_uint32()
            com_call(device, GET_ALLOCATED_STRING, (POINTER(GUID), POINTER(c_wchar_p), POINTER(c_uint32)),
                     byref(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME), byref(name), byref(length))
            names.append(name.value)
            co_task_mem_free(ctypes.cast(name, c_void_p))
            release(device)
        co_task_mem_free(ctypes.cast(devices, c_void_p))
    finally:
        if attributes:
            release(attributes)
        mfplat.MFShutdown()
    return names


if __name__ == "__main__":
    for index, name in enumerate(list_msmf_cameras()):
        print(f"{index}: {name}")
