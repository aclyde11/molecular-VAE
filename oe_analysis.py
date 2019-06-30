from __future__ import print_function
import os
import sys
import argparse

from openeye import oechem
from openeye import oefastrocs

oepy = os.path.join(os.path.dirname(__file__), "..", "python")
sys.path.insert(0, os.path.realpath(oepy))


def main(argv=[__name__]):

    parser = argparse.ArgumentParser()

    # positional arguments retaining backward compatibility
    parser.add_argument('database',
                        help='File containing the database molecules to be search (format not restricted to *.oeb).')
    parser.add_argument('query', default=[], nargs='+',
                        help='File containing the query molecule(s) to be search (format not restricted to *.oeb).')
    parser.add_argument('--nHits', dest='nHits', type=int, default=100,
                        help='Number of hits to return (default = number of database mols).')


    args = parser.parse_args()

    dbname = args.database

    if not oefastrocs.OEFastROCSIsGPUReady():
        oechem.OEThrow.Info("No supported GPU available!")
        return 0

    # set options
    opts = oefastrocs.OEShapeDatabaseOptions()
    opts.SetLimit(args.nHits)
    print("Number of hits set to %u" % opts.GetLimit())

    # read in database
    ifs = oechem.oemolistream()
    if not ifs.open(dbname):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    print("\nOpening database file %s ..." % dbname)
    timer = oechem.OEWallTimer()
    dbase = oefastrocs.OEShapeDatabase()
    moldb = oechem.OEMolDatabase()
    if not moldb.Open(ifs):
        oechem.OEThrow.Fatal("Unable to open '%s'" % dbname)

    dots = oechem.OEThreadedDots(10000, 200, "conformers")
    if not dbase.Open(moldb, dots):
        oechem.OEThrow.Fatal("Unable to initialize OEShapeDatabase on '%s'" % dbname)

    dots.Total()
    print("%f seconds to load database\n" % timer.Elapsed())

    for qfname in args.query:
        # read in query
        qfs = oechem.oemolistream()
        if not qfs.open(qfname):
            oechem.OEThrow.Fatal("Unable to open '%s'" % qfname)

        mcmol = oechem.OEMol()
        if not oechem.OEReadMolecule(qfs, mcmol):
            oechem.OEThrow.Fatal("Unable to read query from '%s'" % qfname)
        qfs.rewind()

        qmolidx = 0
        while oechem.OEReadMolecule(qfs, mcmol):
            moltitle = mcmol.GetTitle()
            if len(moltitle) == 0:
                moltitle = str(qmolidx)

            print("Searching for %s of %s (%s conformers)" % (moltitle, qfname, mcmol.NumConfs()))

            qconfidx = 0
            max_score = 0
            scores_run = 0
            max_scorer_dbase = 0
            for conf in mcmol.GetConfs():

                for score in dbase.GetSortedScores(conf, opts):
                    dbmol = oechem.OEMol()
                    dbmolidx = score.GetMolIdx()
                    if not moldb.GetMolecule(dbmol, dbmolidx):
                        print("Unable to retrieve molecule '%u' from the database" % dbmolidx)
                        continue
                    print(dbmol.GetTitle())
                    max_scorer_dbase = dbmolidx
                    max_score = max(max_score, score.GetColorTanimoto())
                    scores_run += 1
                    #print( "ColorTanimoto", "%.4f" % score.GetColorTanimoto(), dbmolidx, moltitle)
                    break

                qconfidx += 1
            print("Color: ", max_score, scores_run, dbmolidx)
            print("%s conformers processed" % qconfidx)

        qmolidx += 1
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))