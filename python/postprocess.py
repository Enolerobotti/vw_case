import sys
import paraview.simple as pv


def animate(fn):
    casefoam = pv.OpenFOAMReader(FileName=fn)
    pv.Show(casefoam)
    dp = pv.GetDisplayProperties(casefoam)
    dp.SetPropertyWithName('ColorArrayName', ['POINTS', 'U'])
    view=pv.GetActiveView()
    reader = pv.GetActiveSource()
    tsteps = reader.TimestepValues
    annTime = pv.AnnotateTimeFilter(reader)
    pv.Show(annTime)
    pv.Render()
    while True:
        try:
            for t in tsteps:
                view.ViewTime = t
                pv.Render()
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == '__main__':
    filename = [
        "./thin_plate_case/thin_plate_case.foam",
        "./thick_plate_case/thick_plate_case.foam" # not exists yet
    ]
    animate(filename[0])
