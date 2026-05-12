from abaqus import *
from abaqusConstants import *
import regionToolset
import job
import step
import mesh
import interaction
import load
import visualization
import xyPlot
from odbAccess import *
import csv
import os
import random
import time

TARGET_DIR = r"C:\temp\StandardBilayer_Results" # replace with target directory

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

os.chdir(TARGET_DIR)


TOTAL_RUNS = 500
CSV_FILENAME = 'StandardBilayer_Results.csv'
MODEL_NAME_PREFIX = 'StdBilayer_2D_'


MAT_TPU_E = 25.0
MAT_TPU_NU = 0.48
MAT_TPU_ALPHA = 1.5e-4

MAT_PLA_E_TABLE = (
    (2800.0, 0.35, 20.0),
    (350.0, 0.35, 59.9)
)
MAT_PLA_ALPHA_ISO = 4.1e-5


def create_and_run(run_id, length, t_tpu, t_pla, temp_change):
    model_name = '{}{}'.format(MODEL_NAME_PREFIX, run_id)
    job_name = 'Job_Bilayer_{}'.format(run_id)

    total_h = t_tpu + t_pla
    half_L = length / 2.0 

    if model_name in mdb.models:
        del mdb.models[model_name]

    model = mdb.Model(name=model_name)

    # Materials
    mat_tpu = model.Material(name='TPU')
    mat_tpu.Elastic(table=((MAT_TPU_E, MAT_TPU_NU),))
    mat_tpu.Expansion(table=((MAT_TPU_ALPHA,),))

    mat_pla = model.Material(name='PLA')
    mat_pla.Elastic(temperatureDependency=ON, table=MAT_PLA_E_TABLE)

    target_shrinkage = random.uniform(-0.04,-0.02)
    alpha_11 = target_shrinkage / temp_change
    mat_pla.Expansion(type=ORTHOTROPIC, table=((alpha_11, MAT_PLA_ALPHA_ISO, MAT_PLA_ALPHA_ISO),))

    model.HomogeneousSolidSection(name='Section-TPU', material='TPU')
    model.HomogeneousSolidSection(name='Section-PLA', material='PLA')

    # Geometry
    s1 = model.ConstrainedSketch(name='TPU_Sketch', sheetSize=200.0)
    s1.rectangle(point1=(0.0, 0.0), point2=(half_L, t_tpu))
    part_tpu = model.Part(name='TPU_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part_tpu.BaseShell(sketch=s1)

    s2 = model.ConstrainedSketch(name='PLA_Sketch', sheetSize=200.0)
    s2.rectangle(point1=(0.0, t_tpu), point2=(half_L, total_h))
    part_pla = model.Part(name='PLA_Part', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part_pla.BaseShell(sketch=s2)

    # Assembly
    a = model.rootAssembly
    inst_tpu = a.Instance(name='TPU-Inst', part=part_tpu, dependent=ON)
    inst_pla = a.Instance(name='PLA-Inst', part=part_pla, dependent=ON)

    a.InstanceFromBooleanMerge(name='Strip', instances=(inst_tpu, inst_pla),
                               keepIntersections=ON, originalInstances=DELETE, domain=GEOMETRY)

    part = model.parts['Strip']
    faces = part.faces

    # Section assignment
    face_tpu = faces.findAt(((half_L / 2.0, t_tpu / 2.0, 0.0),))
    region_tpu = regionToolset.Region(faces=face_tpu)
    part.SectionAssignment(region=region_tpu, sectionName='Section-TPU')

    face_pla = faces.findAt(((half_L / 2.0, t_tpu + t_pla / 2.0, 0.0),))
    region_pla = regionToolset.Region(faces=face_pla)
    part.SectionAssignment(region=region_pla, sectionName='Section-PLA')

    # Assign Orientation
    part.MaterialOrientation(region=region_pla, orientationType=GLOBAL, axis=AXIS_1,
                             additionalRotationType=ROTATION_NONE)

    # Meshing
    part.seedPart(size=0.2, deviationFactor=0.1, minSizeFactor=0.1)
    elem_type = mesh.ElemType(elemCode=CPE4I, elemLibrary=STANDARD)
    part.setElementType(regions=(faces,), elemTypes=(elem_type,))
    part.generateMesh()

    a.regenerate()
    inst = a.instances['Strip-1']

    # Boundary conditions
    edges = inst.edges
    vertices = inst.vertices

    symm_edge = edges.findAt(((half_L, t_tpu, 0.0),))
    region_symm = regionToolset.Region(edges=symm_edge)
    model.XsymmBC(name='BC-Symm', createStepName='Initial', region=region_symm)

    v_pin = vertices.findAt(((half_L, 0.0, 0.0),))
    model.DisplacementBC(name='BC-Pin', createStepName='Initial',
                         region=regionToolset.Region(vertices=v_pin), u2=0.0)

    v_tip = vertices.findAt(((0.0, total_h, 0.0),))
    a.Set(vertices=v_tip, name='SET-TIP')

    # Steps & loads
    model.StaticStep(name='Step-Heat', previous='Initial', nlgeom=ON)

    region_all = regionToolset.Region(faces=inst.faces)
    model.Temperature(name='T-Init', createStepName='Initial', region=region_all, magnitudes=(0.0,))
    model.Temperature(name='T-Final', createStepName='Step-Heat', region=region_all, magnitudes=(temp_change,))

    # Job
    my_job = mdb.Job(name=job_name, model=model_name, type=ANALYSIS, resultsFormat=ODB)
    my_job.submit()
    my_job.waitForCompletion()

    return job_name, target_shrinkage


def extract_results(job_name):
    odb_name = job_name + '.odb'
    try:
        odb = openOdb(path=odb_name)
        last_frame = odb.steps['Step-Heat'].frames[-1]
        disp_field = last_frame.fieldOutputs['U']
        tip_set = odb.rootAssembly.nodeSets['SET-TIP']
        tip_disp = disp_field.getSubset(region=tip_set)
        deflection = tip_disp.values[0].data[1]
        odb.close()
        return deflection
    except:
        return None

def clean_up_files(job_name):
    time.sleep(0.2)
    extensions = ['.odb', '.dat', '.msg', '.sta', '.com', '.prt', '.log', '.sim', '.stt', '.res', '.lck', '.ipm', '.inp']
    for ext in extensions:
        file_path = job_name + ext
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
                
if __name__ == '__main__':
    with open(CSV_FILENAME, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['Run_ID', 'Length_mm', 'Total_Thick_mm', 'Shrinkage_Strain', 'Deflection_mm'])

        for i in range(1, TOTAL_RUNS + 1):
            p_len = random.uniform(75, 125)
            single_layer_thickness = random.uniform(0.3, 0.5)
            p_t_tpu = single_layer_thickness
            p_t_pla = single_layer_thickness
            
            total_t = p_t_pla + p_t_tpu
            
            total_t = p_t_tpu + p_t_pla
            p_temp = 60.0
            
            try:
                current_job_name, current_shrinkage = create_and_run(i, p_len, p_t_tpu, p_t_pla, p_temp)
                res = extract_results(current_job_name)

                if res is not None:
                    writer.writerow([i, round(p_len, 2), round(total_t, 2), round(current_shrinkage, 4), round(res, 4)])
                    print("Deflection: {:.4f} mm".format(res))
                else:
                    print("   Error: Could not extract results.")
                    
                clean_up_files(current_job_name)

            except Exception as e:
                print("Error in run {}: {}".format(i, e))

    print("\nData saved to {}/{}".format(TARGET_DIR, CSV_FILENAME))