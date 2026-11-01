# coding: ascii
import ScriptEnv

ScriptEnv.Initialize("Ansoft.ElectronicsDesktop")
oDesktop.RestoreWindow()

oProject = oDesktop.GetActiveProject()
if oProject is None:
    oProject = oDesktop.NewProject()

oDesign = oProject.GetActiveDesign()
if oDesign is None:
    oProject.InsertDesign("HFSS", "HFSSDesign1", "HFSS Modal Network", "")
    oDesign = oProject.SetActiveDesign("HFSSDesign1")

oDesign.SetSolutionType(
    "HFSS Modal Network",
    ["NAME:Options", "EnableAutoOpen:=", False]
)

oEditor = oDesign.SetActiveEditor("3D Modeler")
existing_solids = list(oEditor.GetObjectsInGroup("Solids"))
existing_sheets = list(oEditor.GetObjectsInGroup("Sheets"))
if existing_solids or existing_sheets:
    existing_designs = [d.GetName() for d in oProject.GetDesigns()]
    idx = 1
    new_name = "HFSSDesign_Array8"
    while new_name in existing_designs:
        idx += 1
        new_name = "HFSSDesign_Array8_" + str(idx)
    oProject.InsertDesign("HFSS", new_name, "HFSS Modal Network", "")
    oDesign = oProject.SetActiveDesign(new_name)
    oDesign.SetSolutionType(
        "HFSS Modal Network",
        ["NAME:Options", "EnableAutoOpen:=", False]
    )
    oEditor = oDesign.SetActiveEditor("3D Modeler")

oEditor.SetModelUnits(
    ["NAME:Units Parameter", "Units:=", "mm", "Rescale:=", False]
)
oBoundary = oDesign.GetModule("BoundarySetup")
oMesh = oDesign.GetModule("MeshSetup")


def mm(v):
    return "%.12gmm" % float(v)


def quoted(name):
    return chr(34) + str(name) + chr(34)


def make_cylinder(name, x, y, z, radius, height, axis, mat, num_sides="8"):
    solve_inside = False if str(mat).lower() == "pec" else True
    oEditor.CreateCylinder(
        [
            "NAME:CylinderParameters",
            "XCenter:=", mm(x),
            "YCenter:=", mm(y),
            "ZCenter:=", mm(z),
            "Radius:=", mm(radius),
            "Height:=", mm(height),
            "WhichAxis:=", axis,
            "NumSides:=", str(num_sides)
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(143 175 143)",
            "Transparency:=", 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", quoted(mat),
            "SolveInside:=", solve_inside,
            "IsMaterialEditable:=", True
        ]
    )


def make_rect(name, x, y, z, w, h):
    oEditor.CreateRectangle(
        [
            "NAME:RectangleParameters",
            "IsCovered:=", True,
            "XStart:=", mm(x),
            "YStart:=", mm(y),
            "ZStart:=", mm(z),
            "Width:=", mm(w),
            "Height:=", mm(h),
            "WhichAxis:=", "Z"
        ],
        [
            "NAME:Attributes",
            "Name:=", name,
            "Flags:=", "",
            "Color:=", "(255 128 0)",
            "Transparency:=", 0.4,
            "PartCoordinateSystem:=", "Global",
            "UDMId:=", "",
            "MaterialValue:=", quoted("vacuum"),
            "SolveInside:=", True,
            "IsMaterialEditable:=", True
        ]
    )


def assign_port(rect_name, port_name, x_offset):
    oBoundary.AssignLumpedPort(
        [
            "NAME:" + port_name,
            "Objects:=", [rect_name],
            "RenormalizeAllTerminals:=", True,
            "DoDeembed:=", False,
            [
                "NAME:Modes",
                [
                    "NAME:Mode1",
                    "ModeNum:=", 1,
                    "UseIntLine:=", True,
                    [
                        "NAME:IntLine",
                        "Coordinate System:=", "Global",
                        "Start:=", [mm(x_offset), "-0.5mm", "0mm"],
                        "End:=", [mm(x_offset), "0.5mm", "0mm"]
                    ],
                    "AlignmentGroup:=", 0,
                    "CharImp:=", "Zpi",
                    "RenormImp:=", "50ohm"
                ]
            ],
            "ShowReporterFilter:=", False,
            "ReporterFilter:=", [True],
            "Impedance:=", "50ohm"
        ]
    )


arm_names = []
for n in range(8):
    x = n * 62.5
    i = str(n + 1)
    lo_name = "Arm" + i + "_lower"
    hi_name = "Arm" + i + "_upper"
    rc_name = "Port" + i + "_rect"
    make_cylinder(lo_name, x, -30.0, 0.0, 1.0, 29.5, "Y", "pec", "8")
    make_cylinder(hi_name, x,   0.5, 0.0, 1.0, 29.5, "Y", "pec", "8")
    make_rect(rc_name, x - 0.6, -0.5, 0.0, 1.2, 1.0)
    assign_port(rc_name, "Port" + i, x)
    arm_names.extend([lo_name, hi_name])

oEditor.CreateBox(
    [
        "NAME:BoxParameters",
        "XPosition:=", "-35mm",
        "YPosition:=", "-65mm",
        "ZPosition:=", "-35mm",
        "XSize:=", "507.5mm",
        "YSize:=", "130mm",
        "ZSize:=", "70mm"
    ],
    [
        "NAME:Attributes",
        "Name:=", "AirBox",
        "Flags:=", "",
        "Color:=", "(128 128 255)",
        "Transparency:=", 0.9,
        "PartCoordinateSystem:=", "Global",
        "UDMId:=", "",
        "MaterialValue:=", quoted("vacuum"),
        "SolveInside:=", True,
        "IsMaterialEditable:=", True
    ]
)

oBoundary.AssignRadiation(
    [
        "NAME:Rad1",
        "Objects:=", ["AirBox"],
        "IsFssReference:=", False,
        "IsForPML:=", False
    ]
)

oMesh.InitialMeshSettings(
    [
        "NAME:MeshSettings",
        [
            "NAME:GlobalSurfApproximation",
            "CurvedSurfaceApproxChoice:=", "UseSlider",
            "SliderMeshSettings:=", 5
        ],
        [
            "NAME:GlobalCurvilinear",
            "Apply:=", False
        ],
        [
            "NAME:GlobalModelRes",
            "UseAutoLength:=", True
        ],
        "MeshMethod:=", "AnsoftClassic",
        "UseLegacyFaceterForTauVolumeMesh:=", False,
        "DynamicSurfaceResolution:=", False,
        "UseFlexMeshingForTAUvolumeMesh:=", False,
        "UseAlternativeMeshMethodsAsFallBack:=", True,
        "AllowPhiForLayeredGeometry:=", True
    ]
)

oMesh.AssignLengthOp(
    [
        "NAME:AirMesh",
        "RefineInside:=", True,
        "Enabled:=", True,
        "Objects:=", ["AirBox"],
        "RestrictElem:=", True,
        "NumMaxElem:=", "20000",
        "RestrictLength:=", True,
        "MaxLength:=", "40mm"
    ]
)

oMesh.AssignLengthOp(
    [
        "NAME:DipoleSurfMesh",
        "RefineInside:=", False,
        "Enabled:=", True,
        "Objects:=", arm_names,
        "RestrictElem:=", True,
        "NumMaxElem:=", "4000",
        "RestrictLength:=", True,
        "MaxLength:=", "6mm"
    ]
)

oAnalysis = oDesign.GetModule("AnalysisSetup")
oAnalysis.InsertSetup(
    "HfssDriven",
    [
        "NAME:Setup1",
        "SolveType:=", "Single",
        "Frequency:=", "2.4GHz",
        "MaxDeltaS:=", 0.02,
        "PortsOnly:=", False,
        "UseMatrixConv:=", False,
        "MaximumPasses:=", 10,
        "MinimumPasses:=", 2,
        "MinimumConvergedPasses:=", 1,
        "PercentRefinement:=", 15,
        "IsEnabled:=", True,
        "BasisOrder:=", 1,
        "DoLambdaRefine:=", True,
        "DoMaterialLambda:=", True,
        "SetLambdaTarget:=", False,
        "PortAccuracy:=", 2,
        "SaveAnyFields:=", True,
        "SaveRadFieldsOnly:=", False
    ]
)

oRadField = oDesign.GetModule("RadField")
oRadField.InsertInfiniteSphereSetup(
    [
        "NAME:FarField1",
        "UseCustomRadiationSurface:=", False,
        "CSDefinition:=", "Theta-Phi",
        "Polarization:=", "Linear",
        "ThetaStart:=", "-90deg",
        "ThetaStop:=", "90deg",
        "ThetaStep:=", "1deg",
        "PhiStart:=", "0deg",
        "PhiStop:=", "0deg",
        "PhiStep:=", "1deg",
        "UseLocalCS:=", False
    ]
)

oDesktop.AddMessage("", "", 0, "Student-safe 8-element 2.4 GHz dipole array created.")
print("Done.")