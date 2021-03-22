#!/bin/env python
import pandas
import sys
from philipsCsvAdapter import philipsCsvAdapter
from geCsvAdapter import geCsvAdapter


class UpdateCsv:
    def __init__(self, csvName, positionParam):
        self.csvName = csvName
        self.positionParam = positionParam
        self.df = None

    def __del__(self):
        print "[Success] csv has been update succefully"

    def line_prepender(self, filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            if content[0] == 'A' and content[1] == ',':
                print "[Header] header already existing.."
                return
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)
        return

    def addHeader(self):
        self.line_prepender(self.positionParam["file"], "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ,AK,AL,AM\n")
        return

    def openCsv(self):
        self.addHeader()
        self.df = pandas.read_csv(self.csvName, decimal=',')

    def run(self):
        if (self.positionParam["type"] == "philips"):
            run = philipsCsvAdapter(self.df, self.positionParam)
        elif (self.positionParam["type"] == "ge"):
            run = geCsvAdapter(self.df, self.positionParam)
        self.df = run.updateColumn()
        self.df.to_csv(self.positionParam["output"], header=False, index=False, decimal='.')


if __name__ == "__main__":
    nb = len(sys.argv)
    usage = "[Usage] -h [Help] -t [philips|ge] -lo [position longitudinal] -la [position lateral] -ht [hauteur de table] -f [file.csv] -o [output.csv]"

    ctx = 1
    arg = dict({'lo' : 0, 'la' : 0, 'ht' : 0, 'file': "export91withHeader.csv", 'output' : 'UpdatedExport.csv', 'type' : 'philips'})
    if nb > 2:
        while ctx < nb:
            if sys.argv[ctx] == "-h" or sys.argv[ctx] == "--help":
                print usage
                exit(0)
            elif sys.argv[ctx] == "-lo":
                arg["lo"] = sys.argv[ctx + 1]
            elif sys.argv[ctx] == "-la":
                arg["la"] = sys.argv[ctx + 1]
            elif sys.argv[ctx] == "-ht":
                arg["ht"] = sys.argv[ctx + 1]
            elif sys.argv[ctx] == "-f":
                arg["file"] = sys.argv[ctx + 1]
            elif sys.argv[ctx] == "-o":
                arg["output"] = sys.argv[ctx + 1]
            elif sys.argv[ctx] == "-t" or sys.argv[ctx] == "--type":
                arg["type"] = sys.argv[ctx + 1]
            ctx += 1
    else:
        print usage
        exit(0)

    update = UpdateCsv(arg["file"], arg)
    update.openCsv()
    update.run()
