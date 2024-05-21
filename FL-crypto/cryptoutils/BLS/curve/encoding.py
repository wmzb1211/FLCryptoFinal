"""
    Functions to encode and decode data to base64.
    Improves readability of the files.
    Implements the ZCash serialization standard, except
    for the representation of finite field elements:
    https://tools.ietf.org/html/draft-irtf-cfrg-pairing-friendly-curves-09#appendix-C

    "
        At a high level, the serialization format is defined as follows:

       *  Serialized points include three metadata bits that indicate
          whether a point is compressed or not, whether a point is the point
          at infinity or not, and (for compressed points) the sign of the
          point's y-coordinate.

       *  Points on E are serialized into 48 bytes (compressed) or 96 bytes
          (uncompressed).  Points on E' are serialized into 96 bytes
          (compressed) or 192 bytes (uncompressed).

       *  The serialization of a point at infinity comprises a string of
          zero bytes, except that the metadata bits may be nonzero.

       *  The serialization of a compressed point other than the point at
          infinity comprises a serialized x-coordinate.

       *  The serialization of an uncompressed point other than the point at
          infinity comprises a serialized x-coordinate followed by a
          serialized y-coordinate.
    "

"""

import base64

from .curve import (
    curve_order,
    FQ,
    FQ2,
    BLS12_381_FQ as BaseCurve,
    BLS12_381_FQ2 as ExtCurve
)


#### Encoding properties ####

ENDIANNESS = 'big'
#7th bit of the first byte
BIT_COMPRESSED = 1<<7
BIT_INF = 1<<6
BIT_SIGN = 1<<5

#set to falso if you dont want to use compression
POINT_COMPRESSION = True

#its not a curve point, just an integer
PRIVKEY_SIZE = 32
#size of FQ element
FQ_SIZE = 48

#size depends on if compression is enabled or not
def PUBKEY_SIZE(comp):
    return FQ_SIZE*(1 if comp else 2)

def SIGNATURE_SIZE(comp):
    return 2*FQ_SIZE*(1 if comp else 2)

#return the sign of a FQ point
def sign_F(e):
    return 1 if e.val > (FQ.order()-1)//2 else 0


#return the sign of a FQ2 point
def sign_F2(e):
    if e.val[1] == 0:
        return sign_F(e.val[0])
    elif e.val[1] > (FQ2.char()-1)//2:
        return 1
    else:
        return 0

#build metadata bits from its value (compression, infinite, sign)
def metadata_bits(c,i,s):
    return c*BIT_COMPRESSED + i*BIT_INF + s*BIT_SIGN

def is_comp(b):
    return b & BIT_COMPRESSED != 0

def is_inf(b):
    return b & BIT_INF != 0

def sign(b):
    return (b & BIT_SIGN) // BIT_SIGN


# Returns the public key encoded in base64
def encodePubKey(pk):
    x, y = pk

    if pk.is_infinite():
        x_bytes = bytes(FQ_SIZE)
    else:
        x_bytes = x.val.to_bytes(FQ_SIZE, byteorder=ENDIANNESS)

    if POINT_COMPRESSION:
        y_bytes = b''
    elif pk.is_infinite():
        y_bytes = bytes(FQ_SIZE)
    else:
        y_bytes = y.val.to_bytes(FQ_SIZE, byteorder=ENDIANNESS)

    c_bit = 1 if POINT_COMPRESSION else 0
    i_bit = 1 if pk.is_infinite() else 0
    if c_bit == 0 or i_bit == 1:
        s_bit = 0
    else:
        s_bit = sign_F(y)

    first_byte = x_bytes[0] + metadata_bits(c_bit, i_bit, s_bit)

    tot_bytes = bytes([first_byte]) + x_bytes[1:] + y_bytes
    return base64.b64encode(tot_bytes)

# Returns the public key decoded from base64
def decodePubKey(pkStr):
    byte = base64.b64decode(pkStr)

    if len(byte) < 2:
        raise ValueError("It seems like your public key file is corrupted")

    metadata = byte[0]
    byte = bytes([byte[0] & ((1<<5)-1)]) + byte[1:]

    if len(byte) != PUBKEY_SIZE(is_comp(metadata)):
        raise ValueError("It seems like your public key file is corrupted")

    if is_inf(metadata):
        return BaseCurve()

    if is_comp(metadata):
        x = FQ(int.from_bytes(byte, byteorder=ENDIANNESS))
        
        y2 = x**3 + 4
        if not y2.is_quadratic():
            raise ValueError("It seems like your public key file is corrupted")
        y = y2.sqrt()[sign(metadata)]

    else:
        x = FQ(int.from_bytes(byte[:FQ_SIZE], byteorder=ENDIANNESS))
        y = FQ(int.from_bytes(byte[FQ_SIZE:], byteorder=ENDIANNESS))

    try:
        return BaseCurve(x,y)
    except ValueError as e:
        raise ValueError("It seems your public key file is corrupted:,", e)


# Returns the private key encoded in base64
def encodePrivKey(sk):
    return base64.b64encode(sk.to_bytes(PRIVKEY_SIZE, byteorder=ENDIANNESS))

# Returns the private key decoded from base64
def decodePrivKey(skStr):
    byte = base64.b64decode(skStr) 
    if len(byte) != PRIVKEY_SIZE:
        raise ValueError("It seems like your private key file is corrupted.")

    res = int.from_bytes(byte, byteorder=ENDIANNESS)
    if res < 0 or res > curve_order:
        raise ValueError("It seems like your private key file is corrupted.")
    return res


# Returns the signature encoded in base64
def encodeSignature(sig):
    x, y = sig

    if sig.is_infinite():
        x_bytes = bytes(2*FQ_SIZE)
    else:
        x0, x1 = x.val[0].val, x.val[1].val
        x_bytes = (x0.to_bytes(FQ_SIZE, byteorder=ENDIANNESS)
                    + x1.to_bytes(FQ_SIZE, byteorder=ENDIANNESS))

    if POINT_COMPRESSION:
        y_bytes = b''
    elif sig.is_infinite():
        y_bytes = bytes(2*FQ_SIZE)
    else:
        y0, y1 = y.val[0].val, y.val[1].val
        y_bytes = (y0.to_bytes(FQ_SIZE, byteorder=ENDIANNESS)
                    + y1.to_bytes(FQ_SIZE, byteorder=ENDIANNESS))

    c_bit = 1 if POINT_COMPRESSION else 0
    i_bit = 1 if sig.is_infinite() else 0
    if c_bit == 0 or i_bit == 1:
        s_bit = 0
    else:
        s_bit = sign_F2(y)


    first_byte = x_bytes[0] + metadata_bits(c_bit, i_bit, s_bit)

    tot_bytes = bytes([first_byte]) + x_bytes[1:] + y_bytes
    return base64.b64encode(tot_bytes)



# Returns the signature decoded from base64
def decodeSignature(sigStr):
    byte = base64.b64decode(sigStr)

    if len(byte) < 2:
        raise ValueError("It seems like the signature file is corrupted")

    metadata = byte[0]
    byte = bytes([byte[0] & ((1<<5)-1)]) + byte[1:]

    if len(byte) != SIGNATURE_SIZE(is_comp(metadata)):
        raise ValueError("It seems like the signature file is corrupted")

    if is_inf(metadata):
        return ExtCurve()

    x0 = int.from_bytes(byte[:FQ_SIZE], byteorder=ENDIANNESS)
    x1 = int.from_bytes(byte[FQ_SIZE:2*FQ_SIZE], byteorder=ENDIANNESS)
    x = FQ2([x0, x1])

    if is_comp(metadata):
        y2 = x**3 + FQ2([4,4])
        if not y2.is_quadratic():
            raise ValueError("It seems like the signature file is corrupted")
        ys = y2.sqrt()
        if sign(metadata) == sign_F2(ys[0]):
            y = ys[0]
        else:
            y = ys[1]


    else:
        y0 = int.from_bytes(byte[2*FQ_SIZE:3*FQ_SIZE], byteorder=ENDIANNESS)
        y1 = int.from_bytes(byte[3*FQ_SIZE:], byteorder=ENDIANNESS)
        y = FQ2([y0, y1])

    try:
        return ExtCurve(x,y)
    except ValueError as e:
        raise ValueError("It seems like the signature file is corrupted:,", e)

