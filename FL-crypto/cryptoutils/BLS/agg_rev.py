from .curve.curve import (
    curve_order,
    G1,
    G2,
    # pairing,
    BLS12_381_FQ2
)
from .curve.pairing import pairing
from .curve.encoding import (
    ENDIANNESS
)
import random
import time
from hashlib import sha256


def hashToPoint(message):
    # TODO secure hashing function

    hint = int.from_bytes(sha256(message).digest(), byteorder=ENDIANNESS)

    h = hint % curve_order

    return G2 * h


# Generates both public and secret keys
def keyGen():
    sk = random.randint(0, curve_order)
    pk = G1 * sk
    pk2 = G2 * sk
    return pk, pk2, sk


# Generates a signature of a file
def sign(message, privKey):
    H = hashToPoint(message)
    t1 = time.time()
    signature = privKey * H
    t2 = time.time()
    # print(t2-t1)
    return signature


def aggregate_sign(sigs):
    aggSig = sum(sigs,BLS12_381_FQ2())
    return aggSig


def aggregate_verify(hashs,aggSig,pubKeys,pubKey2s,alphas):
    # start = time.time()
    p1 = pairing(pubKeys[0],hashs[0]+alphas[0]*G2)
    for H,pubKey,alpha in zip(hashs[1:],pubKeys[1:],alphas[1:]):
        p1 *= pairing(pubKey,H+alpha*G2)
    for alpha,pubKey2 in zip(alphas,pubKey2s):
        aggSig += alpha*pubKey2
    p2 = pairing(G1,aggSig)
    # end = time.time()
    # print('verifying time is :')
    return p1 == p2

def subgroup_aggregate_verify(Hs, aggSign, pubKeys, pubKey2s, alphas):
    p1 = pairing(pubKeys[0], Hs[0] + alphas[0] * G2)
    for H, pubKey, alpha in zip(Hs[1:], pubKeys[1:], alphas[1:]):
        p1 *= pairing(pubKey, H + alpha * G2)
    for alpha, pubKey2 in zip(alphas, pubKey2s):
        aggSign += alpha * pubKey2
    p2 = pairing(G1, aggSign)
    return p1 == p2