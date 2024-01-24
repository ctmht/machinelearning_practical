import io


def load(ident: str, target: str, intype: str= 'r', suff: str= "")\
        -> (io.FileIO, io.FileIO):
    """ Get text and labels files """
    tpath = ident + target + "_text" + suff + ".txt"
    lpath = ident + target + "_labels" + suff + ".txt"

    print("\t  Text file: " + tpath)
    print("\tLabels file: " + lpath)

    tfile = open(tpath, intype, encoding="utf-8")
    lfile = open(lpath, intype, encoding="utf-8")

    return tfile, lfile