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

print("Starting single thermal deformation run for PLA/TPU strip...")

# =======================================================
# CONFIGURATION
# =======================================================
MODEL_NAME = 'Thermal_Bimetal_Single'
JOB_NAME = 'Job_PLA_TPU_Heat'

# Geometry Parameters (mm)
LENGTH = 100.0
T_PLA = 0.5  # Thickness of passive layer (PLA)
T_TPU = 0.5  # Thickness of active layer (TPU)
TOTAL_H = T_PLA + T_TPU

# Thermal Load (°C)
TEMP_CHANGE = 50.0

# Material Properties (Representative Values)
# PLA (Passive)
MAT_PLA_E = 3500.0     # Young's Modulus (MPa)
MAT_PLA_NU = 0.35      # Poisson's Ratio
MAT_PLA_ALPHA = 6.8e-5 # Thermal Expansion Coefficient

# TPU (Active)
MAT_TPU_E = 50.0       # Young's Modulus (MPa)
MAT_TPU_NU = 0.48      # Poisson's Ratio
MAT_TPU_ALPHA = 1.5e-4 # Thermal Expansion Coefficient

# Clean up existing model if it exists
if MODEL_NAME in mdb.models:
    del mdb.models[MODEL_NAME]

model = mdb.Model(name=MODEL_NAME)

# =======================================================
# 1. MATERIALS & SECTIONS
# =======================================================
mat_pla = model.Material(name='PLA')
mat_pla.Elastic(table=((MAT_PLA_E, MAT_PLA_NU),))
mat_pla.Expansion(table=((MAT_PLA_ALPHA,),))

mat_tpu = model.Material(name='TPU')
mat_tpu.Elastic(table=((MAT_TPU_E, MAT_TPU_NU),))
mat_tpu.Expansion(table=((MAT_TPU_ALPHA,),))

model.HomogeneousSolidSection(name='Section-PLA', material='PLA', thickness=1.0)
model.HomogeneousSolidSection(name='Section-TPU', material='TPU', thickness=1.0)

# =======================================================
# 2. GEOMETRY (The Strip)
# =======================================================
s = model.ConstrainedSketch(name='Strip_Profile', sheetSize=200.0)
s.rectangle(point1=(0.0, 0.0), point2=(LENGTH, TOTAL_H))

part = model.Part(name='Strip', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
part.BaseShell(sketch=s)

# Partition the face to create the two layers at Y = T_PLA
f = part.faces
pickedFaces = f.getSequenceFromMask(mask=('[#1 ]',), )
t = part.MakeSketchTransform(sketchPlane=f[0], sketchPlaneSide=SIDE1, origin=(0.0, 0.0, 0.0))
s_part = model.ConstrainedSketch(name='Partition', sheetSize=200.0, transform=t)
s_part.Line(point1=(0.0, T_PLA), point2=(LENGTH, T_PLA))

part.PartitionFaceBySketch(faces=pickedFaces, sketch=s_part)

# =======================================================
# 3. ASSIGN SECTIONS
# =======================================================
faces = part.faces

# Assign PLA (Bottom Face)
face_pla = faces.findAt(((LENGTH / 2.0, T_PLA / 2.0, 0.0),))
region_pla = regionToolset.Region(faces=face_pla)
part.SectionAssignment(region=region_pla, sectionName='Section-PLA', offset=0.0,
                       offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# Assign TPU (Top Face)
face_tpu = faces.findAt(((LENGTH / 2.0, T_PLA + T_TPU / 2.0, 0.0),))
region_tpu = regionToolset.Region(faces=face_tpu)
part.SectionAssignment(region=region_tpu, sectionName='Section-TPU', offset=0.0,
                       offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)

# =======================================================
# 4. MESHING
# =======================================================
part.seedPart(size=1.0, deviationFactor=0.1, minSizeFactor=0.1) # Slightly finer mesh for better visual curve
elem_type = mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD)
part.setElementType(regions=(faces,), elemTypes=(elem_type,))
part.generateMesh()

# =======================================================
# 5. ASSEMBLY & SETS
# =======================================================
a = model.rootAssembly
inst = a.Instance(name='Strip-1', part=part, dependent=ON)

# Set: Fixed End (Left Edge, x=0)
edges = inst.edges
left_edges = edges.findAt(((0.0, T_PLA / 2.0, 0.0),), ((0.0, T_PLA + T_TPU / 2.0, 0.0),))
region_fix = regionToolset.Region(edges=left_edges)

# =======================================================
# 6. STEPS & LOADS
# =======================================================
# =======================================================
# 6. STEPS & LOADS
# =======================================================
# NLGEOM=ON is critical for large thermal bending
model.StaticStep(name='Step-Heat', previous='Initial', nlgeom=ON)

# ---> NEW ADDITION: Tell Abaqus to save the results! <---
model.FieldOutputRequest(name='F-Output-1', createStepName='Step-Heat',
                         variables=('S', 'E', 'U', 'NT'))

# Boundary Condition: Encastre left side
model.EncastreBC(name='BC-Fix', createStepName='Initial', region=region_fix)

# Load: Temperature Field
all_faces = inst.faces
region_all = regionToolset.Region(faces=all_faces)

# Initial Temp (0)
model.Temperature(name='Temp-Init', createStepName='Initial',
                  region=region_all, distributionType=UNIFORM, magnitudes=(0.0,))

# Final Temp (temp_change)
model.Temperature(name='Temp-Final', createStepName='Step-Heat',
                  region=region_all, distributionType=UNIFORM, magnitudes=(TEMP_CHANGE,))
# =======================================================
# 7. JOB
# =======================================================
my_job = mdb.Job(name=JOB_NAME, model=MODEL_NAME, type=ANALYSIS,
                 resultsFormat=ODB, description='Single run for PLA/TPU strip thermal deformation')

print("Submitting job...")
my_job.submit()
my_job.waitForCompletion()

print("Job completed! You can now open {}.odb in the Visualization module.".format(JOB_NAME))