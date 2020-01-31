from openrem.remapp.tools.openskin import geomclass
from openrem.remapp.tools.openskin import geomfunc
from openrem.remapp.tools.openskin import skinMap


class CalcExpMap(object):

    def __init__(self, phantom_type=None, pat_pos=None,
                 pat_mass=73.2, pat_height=178.6,
                 table_thick=0.5, table_width=40.0, table_length=150.0,
                 matt_thick=4.0):

        self.phantom_type = phantom_type
        self.pat_mass = pat_mass
        self.pat_height = pat_height
        self.table_thick = table_thick
        self.table_width = table_width
        self.table_length = table_length
        self.matt_thick = matt_thick
        self.patPos = pat_pos
        self.table_trans = 1

        if self.phantom_type == 'flat':
            # I think that the values passed to geomclass.Phantom below should be parameters
            # rather than hard-written values. Is that correct?
            # def __init__(self, phantomType, origin, width, height, scale):
            # self.phantom = geomclass.Phantom("flat", [025, 0, 0], 50, 150, 1)
            # Where does the 025 come from?
            # The 1 is the scale
            self.phantom = geomclass.PhantomFlat("flat", [25, 0, 0], self.table_width, self.table_length, 1)
            self.matt_thick = 0.0
            
        elif self.phantom_type == "3D":
            self.phantom = geomclass.Phantom3([0, -5, -self.matt_thick], mass=self.pat_mass, height=self.pat_height,
                                              pat_pos=pat_pos)

        self.my_dose = geomclass.SkinDose(self.phantom)
        self.num_views = 0

    def add_view(self,
                 delta_x=None, delta_y=None, delta_z=None,
                 angle_x=None, angle_y=None,
                 d_ref=None, dap=None, ref_ak=None,
                 kvp=None, filter_cu=None,
                 run_type=None, frames=None, end_angle=None, pat_pos=None):
        
        if pat_pos == "FFS" or pat_pos == "ffs":
            delta_x = -delta_x
            delta_y = -delta_y
        elif pat_pos == "HFP" or pat_pos == "hfp":
            delta_z = -delta_z
            delta_x = -delta_x 
        elif pat_pos == "FFP" or pat_pos == "ffp":
            delta_y = -delta_y
            delta_z = -delta_z
        elif pat_pos == "HFS" or pat_pos == "hfs":
            pass
        else:
            print("No orientation known. Quitting skin dose calculator")
            return

        self.my_dose.add_view(str(self.num_views))
        self.num_views += 1

        area = dap / ref_ak * 100. * 100.

        x_ray = geomfunc.build_ray(delta_x, delta_y, delta_z, angle_x, angle_y, d_ref + 15)
        
        if self.phantom.phantomType == "flat":
            self.table_trans = geomfunc.get_table_trans(kvp, filter_cu)
        elif self.phantom.phantomType == "3d":
            self.table_trans = geomfunc.get_table_mattress_trans(kvp, filter_cu)

        if 'Rotational' in run_type:
            self.my_dose.add_dose(skinMap.rotational(x_ray, angle_x, end_angle, int(frames), self.phantom, area, ref_ak,
                                                     kvp, filter_cu, d_ref,
                                                     self.table_length, self.table_width,
                                                     self.table_trans,
                                                     self.table_thick + self.matt_thick))
        else:
            self.my_dose.add_dose(skinMap.skin_map(x_ray, self.phantom, area, ref_ak, kvp, filter_cu, d_ref,
                                                   self.table_length, self.table_width,
                                                   self.table_trans,
                                                   self.table_thick + self.matt_thick))
