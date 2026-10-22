"""Check archived polar coordinate-frame assumptions with bounded fresh XFOIL cases.

These results are diagnostics, not replacement training labels. Original data,
models and reports are read-only. Each solver run has its own new output path.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from robust_airfoil.geometry.parser import parse_coordinate_file
from robust_airfoil.geometry.normalise import normalise_geometry
from robust_airfoil.data.polar_parser import parse_polar_file
from robust_airfoil.hashing import sha256_file
from robust_airfoil.validation.xfoil_runner import XFoilCase, run_xfoil, _run_xfoil_segment

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent/'CONTINUATION_20261020/FRAME_CHECK'
EXE=ROOT/'tools/xfoil/xfoil.exe'
OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for name in ['a18','ag35','naca2412']:
    raw_path=ROOT/'data/aerofoil_data'/f'{name}.dat'
    raw=parse_coordinate_file(raw_path)
    normal=normalise_geometry(raw)
    old=ROOT/'data/robust_v2/raw/airfoiltools_snapshot/repo/dat/case-dat'/f'{name}-il-1000000.csv'
    archive=parse_polar_file(old).points
    for kind,coordinates in [('raw',raw),('normalised',normal.coordinates)]:
        directory=OUT/name/kind
        if directory.exists():
            raise FileExistsError(f'Preserve existing frame diagnostic: {directory}')
        directory.mkdir(parents=True)
        path=directory/'geometry.dat'
        path.write_text('_'+name+'_'+kind+'\n'+'\n'.join(f'{x:.12f} {y:.12f}' for x,y in coordinates)+'\n',encoding='utf-8')
        started=time.monotonic()
        result=run_xfoil(EXE,XFoilCase(path,1e6,0.,9.,0.,5.,1.,200,'positive'),directory,45)
        merged=result.parsed_points.merge(archive,on='alpha_deg',suffixes=('_local','_archive'))
        record={'airfoil':name,'kind':kind,'rotation_deg':float(np.degrees(normal.rotation_radians)),
                'chord':normal.chord,'source_geometry_sha256':sha256_file(raw_path),
                'used_geometry_sha256':sha256_file(path),'executable_sha256':sha256_file(EXE),
                'solver_status':result.status,'return_code':result.return_code,'seconds':time.monotonic()-started,
                'matched_archive_points':len(merged),
                'mean_cl_difference_vs_recorded_alpha':float((merged.cl_local-merged.cl_archive).mean()) if len(merged) else None,
                'cl_rmse_vs_recorded_alpha':float(np.sqrt(np.mean((merged.cl_local-merged.cl_archive)**2))) if len(merged) else None}
        rows.append(record)
        print(json.dumps(record),flush=True)
# A representative full-range continuous run estimates the actual controlled-label workload.
path=OUT/'naca2412/normalised/geometry.dat'
for start,end,step,label in [(0.,16.,.5,'positive'),(0.,-12.,-.5,'negative')]:
    output=OUT/'full_range_timing'/label
    result=_run_xfoil_segment(EXE,XFoilCase(path,1e6,0.,9.,start,end,step,200,label),output,30)
    print('FULL_RANGE',label,result.status,len(result.parsed_points),flush=True)
(OUT/'FRAME_CHECK.json').write_text(json.dumps({'cases':rows,'interpretation':'Diagnostic comparison of recorded-axis and normalised-coordinate solver outcomes. Does not establish all archived generator settings or justify an unrecorded frame transformation.'},indent=2),encoding='utf-8')
print('FRAME_CHECK_COMPLETE',flush=True)
