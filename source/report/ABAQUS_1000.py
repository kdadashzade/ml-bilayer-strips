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
import random
import csv
import os

# CONFIGURATION
# =======================================================
TOTAL_RUNS = 200
CSV_FILENAME = 'Bimetal_Dataset_200Runs.csv'
MODEL_NAME_PREFIX = 'Model_PLA_TPU_'

# Material properties
# TPU - passive, bottom layer
MAT_TPU_E = 25.0  # Modulus (MPa)
MAT_TPU_NU = 0.48  # Poisson's Ratio
MAT_TPU_ALPHA = 6.8e-5  # Thermal Expansion Coefficient

# PLA - active, top layer
MAT_PLA_E = 350.0  # Modulus at Tg (MPa)
MAT_PLA_NU = 0.35  # Poisson's Ratio
MAT_PLA_ALPHA = 1.5e-4  # Thermal Expansion Coefficient


def create_and_run(run_id, length, t_tpu, t_pla, temp_change):
    model_name = '{}{}'.format(MODEL_NAME_PREFIX, run_id)
    job_name = 'Job_Run_{}'.format(run_id)
    total_h = t_tpu + t_pla

    if model_name in mdb.models:
        del mdb.models[model_name]

    model = mdb.Model(name=model_name)

    # 1. MATERIALS & SECTIONS
    mat_tpu = model.Material(name='TPU')
    mat_tpu.Elastic(table=((MAT_TPU_E, MAT_TPU_NU),))
    mat_tpu.Expansion(table=((MAT_TPU_ALPHA,),))

    mat_pla = model.Material(name='PLA')
    mat_pla.Elastic(table=((MAT_PLA_E, MAT_PLA_NU),))
    mat_pla.Expansion(table=((MAT_PLA_ALPHA,),))

    model.HomogeneousSolidSection(name='Section-TPU', material='TPU', thickness=1.0)
    model.HomogeneousSolidSection(name='Section-PLA', material='PLA', thickness=1.0)

    # 2. GEOMETRY
    s = model.ConstrainedSketch(name='Strip_Profile', sheetSize=500.0)
    s.rectangle(point1=(0.0, 0.0), point2=(length, total_h))

    part = model.Part(name='Strip', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)

    # Partition at Y = t_tpu
    f = part.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#1 ]',), )
    t = part.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(0.0, 0.0, 0.0))
    s_part = model.ConstrainedSketch(name='Partition', sheetSize=500.0, transform=t)
    s_part.Line(point1=(0.0, t_tpu), point2=(length, t_tpu))
    part.PartitionFaceBySketch(faces=pickedFaces, sketch=s_part)

    # 3. ASSIGN SECTIONS
    faces = part.faces
    # TPU (Bottom)
    face_tpu = faces.findAt(((length / 2.0, t_tpu / 2.0, 0.0),))
    region_tpu = regionToolset.Region(faces=face_tpu)
    part.SectionAssignment(region=region_tpu, sectionName='Section-TPU', offset=0.0,
                           offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    # PLA (Top)
    face_pla = faces.findAt(((length / 2.0, t_tpu + t_pla / 2.0, 0.0),))
    region_pla = regionToolset.Region(faces=face_pla)
    part.SectionAssignment(region=region_pla, sectionName='Section-PLA', offset=0.0,
                           offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

    # 4. MESHING
    part.seedPart(size=2.0, deviationFactor=0.1, minSizeFactor=0.1)
    elem_type = mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD)
    part.setElementType(regions=(faces,), elemTypes=(elem_type,))
    part.generateMesh()

    # 5. ASSEMBLY & SETS
    a = model.rootAssembly
    inst = a.Instance(name='Strip-1', part=part, dependent=ON)

    # Fixed End (Left Edge, x=0)
    edges = inst.edges
    left_edges = edges.findAt(((0.0, t_tpu / 2.0, 0.0),), ((0.0, t_tpu + t_pla / 2.0, 0.0),))
    region_fix = regionToolset.Region(edges=left_edges)
    a.Set(edges=left_edges, name='Set-Fixed')

    # Tip Node (Right End, Top Corner)
    vertices = inst.vertices
    v_tip = vertices.findAt(((length, total_h, 0.0),))
    a.Set(vertices=v_tip, name='Set-Tip')

    # 6. STEPS & LOADS
    model.StaticStep(name='Step-Heat', previous='Initial', nlgeom=ON)

    # OUTPUT REQUEST - Crucial for extraction!
    model.FieldOutputRequest(name='F-Output-1', createStepName='Step-Heat', variables=('U', 'S', 'E'))

    model.EncastreBC(name='BC-Fix', createStepName='Initial', region=region_fix)

    all_faces = inst.faces
    region_all = regionToolset.Region(faces=all_faces)
    model.Temperature(name='Temp-Init', createStepName='Initial', region=region_all, distributionType=UNIFORM,
                      magnitudes=(0.0,))
    model.Temperature(name='Temp-Final', createStepName='Step-Heat', region=region_all, distributionType=UNIFORM,
                      magnitudes=(temp_change,))

    # 7. JOB
    my_job = mdb.Job(name=job_name, model=model_name, type=ANALYSIS, resultsFormat=ODB)
    my_job.submit()
    my_job.waitForCompletion()

    return job_name


def extract_results(job_name):
    odb_name = job_name + '.odb'
    try:
        odb = openOdb(path=odb_name)
        last_frame = odb.steps['Step-Heat'].frames[-1]
        disp_field = last_frame.fieldOutputs['U']
        tip_set = odb.rootAssembly.nodeSets['SET-TIP']
        tip_disp = disp_field.getSubset(region=tip_set)
        # U2 (vertical displacement) is index 1
        deflection = tip_disp.values[0].data[1]
        odb.close()
        return deflection
    except Exception as e:
        print("Error extracting results for {}: {}".format(job_name, e))
        return None


def clean_files(job_name):
    extensions = ['.odb', '.lck', '.stt', '.com', '.prt', '.sim', '.dat', '.msg', '.sta']
    for ext in extensions:
        fname = job_name + ext
        if os.path.exists(fname):
            try:
                os.remove(fname)
            except:
                pass


# Execution loop
if __name__ == '__main__':
    with open(CSV_FILENAME, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(
            ['Run_ID', 'Length_mm', 'TPU_Thick_mm', 'PLA_Thick_mm', 'PLA_Ratio', 'Temp_Change_C', 'Deflection_mm'])

        for i in range(1, TOTAL_RUNS + 1):

            # Randomize variables
            p_len = random.uniform(100.0, 300.0)
            p_t_tpu = random.uniform(0.2, 1.5)
            p_t_pla = random.uniform(0.2, 1.5)

            ratio = p_t_pla / (p_t_tpu + p_t_pla)  # PLA fraction of total thickness

            p_temp = 60.0

            print("--- Run {}/{} ---".format(i, TOTAL_RUNS))
            print("Params: L={:.1f}mm, TPU={:.2f}mm, PLA={:.2f}mm".format(p_len, p_t_tpu, p_t_pla))

            try:
                job_name = create_and_run(i, p_len, p_t_tpu, p_t_pla, p_temp)
                result = extract_results(job_name)

                if result is not None:
                    writer.writerow([i, round(p_len, 2), round(p_t_tpu, 2), round(p_t_pla, 2), round(ratio, 2), p_temp,
                                     round(result, 4)])
                    print("Deflection: {:.4f} mm".format(result))
                else:
                    print("--> Failed to extract result.")

                clean_files(job_name)

            except Exception as e:
                print("CRITICAL ERROR in Run {}: {}".format(i, e))

    print("Generation complete. Data saved to {}".format(CSV_FILENAME))