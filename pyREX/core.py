# @Author: Arthur Voronin <arthur>
# @Date:   17.04.2021
# @Filename: core.py
# @Last modified by:   arthur
# @Last modified time: 07.05.2021


# core module
from builtins import super
from IPython.display import display
import ipywidgets as wg
import numpy as np
import nglview as ngl
import MDAnalysis as mda
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pyperclip  # copy&paste to clipboard
import threading
import time

# global update for plots
matplotlib.rcParams.update({'font.family': "sans-serif", 'font.weight': "normal", 'font.size': 16})
cp = sns.color_palette()  # access seaborn colors via cp[0] - cp[9]

# TODO: get_color_scheme for any component after checking the lengh of the component (instead of universe?)
# TODO: if if if -> if elif else
"""
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
"""


class iColor(object):
    def __init__(self, iPlayer_cls):
        self._iPlayer = iPlayer_cls
        return

    def get_color_scheme(self, color, universe=None):
        """
        Returns color scheme.

        Args:
            color (str)
            universe (MDA universe/atom grp)

        Returns:
            color_scheme (array): HEX colors for each RES taken from iPlayer.universe (default) or universe.

        Accepted colors (simple):
            'red'       'black'
            'orange'    'grey_80'
            'yellow'    'grey_60'
            'green'     'grey_40'
            'cyan'      'grey_20'
            'blue'      'white'
            'purple'
            'magenta'

        Accepted colors (complex):
            'baw': black and white
            'b/w': black and white
            'rainbow': red to magenta
        """
        color = str(color)

        if universe != None:
            length = len(universe.residues)
        else:
            length = len(self._iPlayer.universe.residues)

        # simple color schemes
        color_scheme_s = {'red': ['0xFF0000'],
                          'orange': ['0xFF9900'],
                          'yellow': ['0xFFFF00'],
                          'green': ['0x00FF00'],
                          'cyan': ['0x00FFFF'],
                          'blue': ['0x0000FF'],
                          'purple': ['0x9900FF'],
                          'magenta': ['0xFF00FF'],

                          'black': ['0x000000'],
                          'grey_80': ['0x444444'],
                          'grey_60': ['0x666666'],
                          'grey_40': ['0x999999'],
                          'grey_20': ['0xCCCCCC'],
                          'white': ['0xFFFFFF']
                          }

        # complex color schemes
        color_scheme_c = {'rainbow': [color_scheme_s['red'],
                                      color_scheme_s['orange'],
                                      color_scheme_s['yellow'],
                                      color_scheme_s['green'],
                                      color_scheme_s['cyan'],
                                      color_scheme_s['blue'],
                                      color_scheme_s['purple'],
                                      color_scheme_s['magenta']],
                          'baw': [color_scheme_s['black'],
                                  color_scheme_s['grey_80'],
                                  color_scheme_s['grey_60'],
                                  color_scheme_s['grey_40'],
                                  color_scheme_s['grey_20'],
                                  color_scheme_s['white']],
                          'b/w': [color_scheme_s['black'],
                                  color_scheme_s['grey_80'],
                                  color_scheme_s['grey_60'],
                                  color_scheme_s['grey_40'],
                                  color_scheme_s['grey_20'],
                                  color_scheme_s['white']]
                          }

        # simple
        if color in color_scheme_s:
            color_scheme = color_scheme_s[color] * length

        # complex
        elif color in color_scheme_c:
            color_scheme = color_scheme_c[color]
            if length % len(color_scheme) == 0:
                repeat = length / len(color_scheme)
            else:
                repeat = 1 + length / len(color_scheme)
            color_scheme = list(np.repeat(color_scheme, repeat))

        # non-existing color schemes
        else:
            print("Color scheme does not exist!")

        return(color_scheme)

    def set_color_scheme(self, color, ci=None):
        """
        Applies color scheme. If ci=None, then target is determinded by <iPlayer_object>.widgets.Components.description.

        Args:
            color (str)
            universe (MDA universe/atom grp)

        Returns:
            color_scheme (array): HEX colors for each RES taken from iPlayer.universe (default) or universe.

        Accepted colors (simple):
            'red'       'black'
            'orange'    'grey_80'
            'yellow'    'grey_60'
            'green'     'grey_40'
            'cyan'      'grey_20'
            'blue'      'white'
            'purple'
            'magenta'

        Accepted colors (complex):
            'baw': black and white
            'b/w': black and white
            'rainbow': red to magenta
        """
        color_schemes = ['red', 'yellow', 'orange', 'green',
                         'cyan', 'blue', 'purple', 'magenta',
                         'black', 'grey_80', 'grey_60', 'grey_40',
                         'grey_20', 'white', 'baw', 'b/w', 'rainbow']
        color_scheme_str = None

        # check for existing color schemes
        if color in color_schemes:
            color_scheme_str = color
            color = self.get_color_scheme(color)

        # update dict / remember color scheme
        if ci == None:
            ci = self._iPlayer.widgets.Components.description.split()[-1]

        if color_scheme_str != None:
            if ci == '0':
                self._iPlayer.widgets._dict_Component_0['Color_Scheme'] = color_scheme_str
            elif ci == '1':
                self._iPlayer.widgets._dict_Component_1['Color_Scheme'] = color_scheme_str
        else:
            if ci == '0':
                self._iPlayer.widgets._dict_Component_0['Color_Scheme'] = color
            elif ci == '1':
                self._iPlayer.widgets._dict_Component_1['Color_Scheme'] = color

        # apply color scheme
        for ri in range(0, 5):
            self._iPlayer.player._set_color_by_residue(color, component_index=ci, repr_index=ri)
        return


class iWidgets(object):
    def __init__(self, iPlayer_cls):
        # align via wg.<type>(..., **.align_kw)
        align_kw = dict(_css=(('.widget-label', 'min-width', '20ex'),),
                        margin='0px 0px 5px 12px')

        # Marie stuff start
        self.YValue = wg.Button(description='', tooltip='click: find global minimum')
        self.FrameTime = wg.Button(description='Time: ' + str(0) + ' ps', tooltip='click: switch between Time and Frame')
        # Marie stuff end

        self.Frame = wg.ToggleButton(value=False, description='Frame: ' + str(0),
                                     button_style='', tooltip='click: copy frame to clipboard', disabled=False)
        self.Time = wg.ToggleButton(value=False, description='Time: ' + str(0) + ' ps',
                                    button_style='', tooltip='click: copy time to clipboard', disabled=False)

        self.Reset = wg.ToggleButton(value=False, description='Reset View', button_style='')
        self.UI = wg.ToggleButton(value=False, description='Hide UI')
        self.PlayerStep = wg.IntSlider(value=1, min=1, max=500, description='Step',
                                       layout=wg.Layout(width='230px'), **align_kw)

        self.Components = wg.ToggleButton(value=False, description='Component 0', button_style='info')
        self._Component_0 = wg.ToggleButton(value=False, description='Component 0', button_style='',
                                            layout=wg.Layout(display='none'))
        self._Component_1 = wg.ToggleButton(value=False, description='Component 1', button_style='',
                                            layout=wg.Layout(display='none'))

        self.Representation = wg.ToggleButton(value=False, description='Representation', button_style='info')
        self._Cartoon = wg.ToggleButton(value=True, description='Cartoon', button_style='',
                                        layout=wg.Layout(display='none'))
        self._BaS = wg.ToggleButton(value=False, description='Ball & Stick', button_style='',
                                    layout=wg.Layout(display='none'))
        self._Surface = wg.ToggleButton(value=False, description='Surface', button_style='',
                                        layout=wg.Layout(display='none'))
        self._Visibility = wg.ToggleButton(value=True, description='Visibility')

        self.Color = wg.ToggleButton(value=False, description='Color', button_style='info')
        self.Color_by_RES = wg.ToggleButton(value=False, description='Color by RES', button_style='')
        self._Color_red = wg.ToggleButton(value=False, description='Red', button_style='',
                                          layout=wg.Layout(display='none'))
        self._Color_orange = wg.ToggleButton(value=False, description='Orange', button_style='',
                                             layout=wg.Layout(display='none'))
        self._Color_yellow = wg.ToggleButton(value=False, description='Yellow', button_style='',
                                             layout=wg.Layout(display='none'))
        self._Color_green = wg.ToggleButton(value=False, description='Green', button_style='',
                                            layout=wg.Layout(display='none'))
        self._Color_cyan = wg.ToggleButton(value=False, description='Cyan', button_style='',
                                           layout=wg.Layout(display='none'))
        self._Color_blue = wg.ToggleButton(value=False, description='Blue', button_style='',
                                           layout=wg.Layout(display='none'))
        self._Color_purple = wg.ToggleButton(value=False, description='Purple', button_style='',
                                             layout=wg.Layout(display='none'))
        self._Color_magenta = wg.ToggleButton(value=False, description='Magenta', button_style='',
                                              layout=wg.Layout(display='none'))
        self._Color_black = wg.ToggleButton(value=False, description='Black', button_style='',
                                            layout=wg.Layout(display='none'))
        self._Color_grey_80 = wg.ToggleButton(value=False, description='Grey 80', button_style='',
                                              layout=wg.Layout(display='none'))
        self._Color_grey_60 = wg.ToggleButton(value=False, description='Grey 60', button_style='',
                                              layout=wg.Layout(display='none'))
        self._Color_grey_40 = wg.ToggleButton(value=False, description='Grey 40', button_style='',
                                              layout=wg.Layout(display='none'))
        self._Color_grey_20 = wg.ToggleButton(value=False, description='Grey 20', button_style='',
                                              layout=wg.Layout(display='none'))
        self._Color_white = wg.ToggleButton(value=False, description='White', button_style='',
                                            layout=wg.Layout(display='none'))
        self._Color_baw = wg.ToggleButton(value=False, description='B/W', button_style='',
                                          layout=wg.Layout(display='none'))
        self._Color_rainbow = wg.ToggleButton(value=False, description='Rainbow', button_style='',
                                              layout=wg.Layout(display='none'))

        # dict: remember representation of component 0
        self._dict_Component_0 = {'Cartoon': True,
                                  'BaS': False,
                                  'Surface': False,
                                  'Visibility': True,
                                  'Color_Scheme': 'rainbow'}
        # dict: remember representation of component 1
        self._dict_Component_1 = {'Cartoon': True,
                                  'BaS': False,
                                  'Surface': False,
                                  'Visibility': True,
                                  'Color_Scheme': 'rainbow'}
        # list: remember last chosen component
        self._track_Components = ['Component 0']

        # make widgets interactive
        def switch_FrameTime(a):
            """
            switch <iPlayer_object>.widgets.FrameTime description beetween:
                - <iPlayer_object>.widgets.Frame.description
                - <iPlayer_object>.widgets.Time.description
            """
            if 'Frame' in self.FrameTime.description:
                self.FrameTime.description = self.Time.description
            else:
                self.FrameTime.description = self.Frame.description
        self.FrameTime.on_click(switch_FrameTime)

        def update_FrameTime(args):
            """
            Update <iPlayer_object>.widgets.FrameTime.description
            """
            frame = args['new']
            time = int(frame * iPlayer_cls.universe.trajectory.dt)  # time in ps
            if 'Frame' in self.FrameTime.description:
                self.FrameTime.description = 'Frame: {}'.format(frame)
            else:
                self.FrameTime.description = 'Time: {} ps'.format(time)
        iPlayer_cls.player.observe(update_FrameTime, 'frame')

        def update_Frame(args):
            """
            Update <iPlayer_object>.widgets.Frame.description
            """
            frame = args['new']
            self.Frame.description = 'Frame: {}'.format(frame)
        iPlayer_cls.player.observe(update_Frame, 'frame')

        def update_Time(args):
            """
            Update <iPlayer_object>.widgets.Time.description
            """
            frame = args['new']
            time = int(frame * iPlayer_cls.universe.trajectory.dt)  # time in ps
            self.Time.description = 'Time: {} ps'.format(time)
        iPlayer_cls.player.observe(update_Time, 'frame')

        def copy_Frame(a):
            """
            Copy frame of <iPlayer_object>.widgets.Frame.description on clickinng the button.
            """
            self.Frame.value = False
            a = str(self.Frame.description.split(': ')[-1])
            pyperclip.copy(a)
        wg.interactive(copy_Frame, a=self.Frame)

        def copy_Time(a):
            """
            Copy time of <iPlayer_object>.widgets.Time.description on clicking the button.
            """
            self.Time.value = False
            a = str(self.Time.description.split(': ')[-1])
            pyperclip.copy(a)
        wg.interactive(copy_Time, a=self.Time)

        def reset_view(a):
            """
            Reset camera view orientation (center+rotate).
            """
            self.Reset.value = False
            # iPlayer_cls.player.center()   #camera orientation is already centered
            iPlayer_cls.player._set_camera_orientation(iPlayer_cls.player._camera_orientation_at_start)
        wg.interactive(reset_view, a=self.Reset)

        def toggle_Visibility(a):
            """
            Toggle Representation -> Visibility of components.
            """
            if self.Components.description == 'Component 0' and a == True:
                iPlayer_cls.player.component_0.show()
            if self.Components.description == 'Component 0' and a == False:
                iPlayer_cls.player.component_0.hide()
            if self.Components.description == 'Component 1' and a == True:
                iPlayer_cls.player.component_1.show()
            if self.Components.description == 'Component 1' and a == False:
                iPlayer_cls.player.component_1.hide()
        wg.interactive(toggle_Visibility, a=self._Visibility)

        def dropdown_Components(a):
            """
            Dropdown menu effect for <iPlayer_object>.widgets.Components button.
            """
            if self.Components.value:
                self._Component_0.layout = wg.Layout(display='visible')
            else:
                self._Component_0.layout = wg.Layout(display='none')
        wg.interactive(dropdown_Components, a=self.Components)
        # mimic layout of _Component_0 button
        wg.jsdlink((self._Component_0, 'layout'), (self._Component_1, 'layout'))

        def dropdown_Representation(a):
            """
            Dropdown menu effect for <iPlayer_object>.widgets.Representations button.
            """
            if self.Representation.value:
                self._Cartoon.layout = wg.Layout(display='visible')
            else:
                self._Cartoon.layout = wg.Layout(display='none')
            return
        wg.interactive(dropdown_Representation, a=self.Representation)
        # mimic layout of Cartoon button
        wg.jsdlink((self._Cartoon, 'layout'), (self._BaS, 'layout'))
        wg.jsdlink((self._Cartoon, 'layout'), (self._Surface, 'layout'))
        wg.jsdlink((self._Cartoon, 'layout'), (self._Visibility, 'layout'))

        def dropdown_Color(a):
            """
            Dropdown menu effect for <iPlayer_object>.widgets.Color button.
            """
            if self.Color.value:
                self._Color_red.layout = wg.Layout(display='visible')
            else:
                self._Color_red.layout = wg.Layout(display='none')
            return
        wg.interactive(dropdown_Color, a=self.Color)
        # mimic layout of Color_red button
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_orange, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_yellow, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_green, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_cyan, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_blue, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_purple, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_magenta, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_black, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_grey_80, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_grey_60, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_grey_40, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_grey_20, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_white, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_baw, 'layout'))
        wg.jsdlink((self._Color_red, 'layout'), (self._Color_rainbow, 'layout'))

        # toggle UI (method is defined below in iWidgets class)
        # mimic layout of Hide/Show UI button
        wg.jsdlink((self.Representation, 'layout'), (self.Color, 'layout'))
        wg.jsdlink((self.Representation, 'layout'), (self.Color_by_RES, 'layout'))
        wg.jsdlink((self.Representation, 'layout'), (self.Components, 'layout'))
        wg.interactive(self.toggle_UI, a=self.UI)

        def update_PlayerStep(step):
            """
            Update step/speed of <iPlayer_object>.widgets.PlayerStep.
            """
            iPlayer_cls.player.player.step = step
            return
        wg.interactive(update_PlayerStep, step=self.PlayerStep)

        def update_Components(a):
            """
            Update <iPlayer_object>.widgets.Components.
            """
            if self._Component_0.value:
                self._Component_0.value = False
                self.Components.description = 'Component 0'
            elif self._Component_1.value:
                self._Component_1.value = False
                self.Components.description = 'Component 1'

            if self.Components.description == 'Component 0'\
                    and self.Components.description != self._track_Components[-1]:

                # set dict values
                self._dict_Component_1['Cartoon'] = self._Cartoon.value
                self._dict_Component_1['BaS'] = self._BaS.value
                self._dict_Component_1['Surface'] = self._Surface.value
                self._dict_Component_1['Visibility'] = self._Visibility.value

                # load dict values
                self._track_Components.append(self.Components.description)
                self._Cartoon.value = self._dict_Component_0['Cartoon']
                self._BaS.value = self._dict_Component_0['BaS']
                self._Surface.value = self._dict_Component_0['Surface']
                self._Visibility.value = self._dict_Component_0['Visibility']

            elif self.Components.description == 'Component 1'\
                    and self.Components.description != self._track_Components[-1]:

                # set dict values
                self._dict_Component_0['Cartoon'] = self._Cartoon.value
                self._dict_Component_0['BaS'] = self._BaS.value
                self._dict_Component_0['Surface'] = self._Surface.value
                self._dict_Component_0['Visibility'] = self._Visibility.value

                # load dict values
                self._track_Components.append(self.Components.description)
                self._Cartoon.value = self._dict_Component_1['Cartoon']
                self._BaS.value = self._dict_Component_1['BaS']
                self._Surface.value = self._dict_Component_1['Surface']
                self._Visibility.value = self._dict_Component_1['Visibility']

            else:  # todo: Distances code
                pass
            return
        wg.interactive(update_Components, a=self._Component_0)
        wg.interactive(update_Components, a=self._Component_1)

        def update_Representation(Cartoon, BaS, Surface):
            """
            Updates representation via add/remove command.
            Colors the representation by looking up the color scheme in hidden dictionary.

            Args:
                Cartoon (bool)
                BaS (bool)
                Surface (bool)
            """
            if iPlayer_cls.widgets.Components.description == 'Component 0':
                iPlayer_cls.player.component_0.clear_representations()
                if Cartoon:
                    iPlayer_cls.player.component_0.add_cartoon(selection="protein rna", color="green")
                if BaS:
                    iPlayer_cls.player.component_0.add_ball_and_stick(selection="all")
                if Surface:
                    iPlayer_cls.player.component_0.add_surface(selection="protein rna", color='blue',
                                                               wireframe=True, opacity=0.2, isolevel=3.)
                cs = iPlayer_cls.widgets._dict_Component_0['Color_Scheme']
                iPlayer_cls.color.set_color_scheme(cs)

            if iPlayer_cls.widgets.Components.description == 'Component 1':
                iPlayer_cls.player.component_1.clear_representations()
                if Cartoon:
                    iPlayer_cls.player.component_1.add_cartoon(selection="protein rna", color="green")
                if BaS:
                    iPlayer_cls.player.component_1.add_ball_and_stick(selection="all")
                if Surface:
                    iPlayer_cls.player.component_1.add_surface(selection="protein rna", color='blue',
                                                               wireframe=True, opacity=0.2, isolevel=3.)
                cs = iPlayer_cls.widgets._dict_Component_1['Color_Scheme']
                iPlayer_cls.color.set_color_scheme(cs)

            return
        wg.interactive(update_Representation, Cartoon=self._Cartoon, BaS=self._BaS,
                       Surface=self._Surface)

        def update_Color(a):
            """
            Update color of representation.
            """
            # simple color schemes
            if self._Color_red.value:
                self._Color_red.value = False
                iPlayer_cls.color.set_color_scheme('red')
            if self._Color_orange.value:
                self._Color_orange.value = False
                iPlayer_cls.color.set_color_scheme('orange')
            if self._Color_yellow.value:
                self._Color_yellow.value = False
                iPlayer_cls.color.set_color_scheme('yellow')
            if self._Color_green.value:
                self._Color_green.value = False
                iPlayer_cls.color.set_color_scheme('green')
            if self._Color_cyan.value:
                self._Color_cyan.value = False
                iPlayer_cls.color.set_color_scheme('cyan')
            if self._Color_blue.value:
                self._Color_blue.value = False
                iPlayer_cls.color.set_color_scheme('blue')
            if self._Color_purple.value:
                self._Color_purple.value = False
                iPlayer_cls.color.set_color_scheme('purple')
            if self._Color_magenta.value:
                self._Color_magenta.value = False
                iPlayer_cls.color.set_color_scheme('magenta')
            if self._Color_black.value:
                self._Color_black.value = False
                iPlayer_cls.color.set_color_scheme('black')
            if self._Color_grey_80.value:
                self._Color_grey_80.value = False
                iPlayer_cls.color.set_color_scheme('grey_80')
            if self._Color_grey_60.value:
                self._Color_grey_60.value = False
                iPlayer_cls.color.set_color_scheme('grey_60')
            if self._Color_grey_40.value:
                self._Color_grey_40.value = False
                iPlayer_cls.color.set_color_scheme('grey_40')
            if self._Color_grey_20.value:
                self._Color_grey_20.value = False
                iPlayer_cls.color.set_color_scheme('grey_20')
            if self._Color_white.value:
                self._Color_white.value = False
                iPlayer_cls.color.set_color_scheme('white')

            # complex color schemes
            if self._Color_baw.value:
                self._Color_baw.value = False
                iPlayer_cls.color.set_color_scheme('b/w')
            if self._Color_rainbow.value:
                self._Color_rainbow.value = False
                iPlayer_cls.color.set_color_scheme('rainbow')
            return
        wg.interactive(update_Color, a=self._Color_red)
        wg.interactive(update_Color, a=self._Color_orange)
        wg.interactive(update_Color, a=self._Color_yellow)
        wg.interactive(update_Color, a=self._Color_green)
        wg.interactive(update_Color, a=self._Color_cyan)
        wg.interactive(update_Color, a=self._Color_blue)
        wg.interactive(update_Color, a=self._Color_purple)
        wg.interactive(update_Color, a=self._Color_magenta)
        wg.interactive(update_Color, a=self._Color_black)
        wg.interactive(update_Color, a=self._Color_grey_80)
        wg.interactive(update_Color, a=self._Color_grey_60)
        wg.interactive(update_Color, a=self._Color_grey_40)
        wg.interactive(update_Color, a=self._Color_grey_20)
        wg.interactive(update_Color, a=self._Color_white)
        wg.interactive(update_Color, a=self._Color_baw)
        wg.interactive(update_Color, a=self._Color_rainbow)

        def click_Color_by_RES(a):
            """
            Color by RES / Apply 'rainbow' color scheme on representation.
            """
            self.Color_by_RES.value = False
            iPlayer_cls.color.set_color_scheme('rainbow')
        wg.interactive(click_Color_by_RES, a=self.Color_by_RES)
        return

    def __repr__(self):
        return "<iWidgets Class>"

    def toggle_UI(self, a=None):
        """
        Toggle UI with a=True/False. If a=None, then UI will switch to other state.
        """
        if a == True and self.UI.value == False:
            self.UI.description = 'True'
        elif a == False and self.UI.value == False:
            self.UI.description = 'False'

        if self.UI.description == 'Show UI' or self.UI.description == 'True':
            self.UI.value = False
            self.UI.description = 'Hide UI'
            self.Components.layout = wg.Layout(width='148px', display='visible')
            self.Representation.layout = wg.Layout(display='visible')

        elif self.UI.description == 'Hide UI' or self.UI.description == 'False':
            self.UI.value = False
            self.UI.description = 'Show UI'
            self.Components.layout = wg.Layout(display="none")
            self.Representation.layout = wg.Layout(display='none')
            self._Cartoon.layout = wg.Layout(display='none')
            self._Color_red.layout = wg.Layout(display='none')
            self._Component_0.layout = wg.Layout(display='none')

            # change values of dropdown menus after toggle
            self.Components.value = False
            self.Representation.value = False
            self.Color.value = False
        return


class iPlayer(object):
    def __init__(self, universe=None):
        """
        init iPlayer

        if universe is PDB ID (str with len = 4) -> fetch online
        if universe is path (str with len > 4) -> create universe
        if universe is universe -> pass universe
        """
        # case 1: input is PDB ID -> fetch online
        if type(universe) is str and len(universe) == 4:
            self.universe = mda.fetch_mmtf(universe)
            self.player = ngl.show_mdanalysis(self.universe)
        # case 2: input is path -> create MDA Universe
        elif type(universe) is str and len(universe) > 4:
            self.universe = mda.Universe(universe)
            self.player = ngl.show_mdanalysis(self.universe)
        # case 3: input is MDA Universe
        else:
            self.universe = universe
            self.player = ngl.show_mdanalysis(self.universe)
        self.player._camera_orientation_at_start = None
        self.color = iColor(self)
        self.widgets = iWidgets(self)
        self.widgets.toggle_UI(False)
        self._init_Representation()
        return

    def __call__(self, layout='default'):
        """
        Show trajectory viewer with GUI.

        Args:
            layout (str):
                'default': default layout
                'every other str': non-default layout

        Alternative:
            Execute show_player() method by calling the object.

        Example:
            tv = core.iPlayer(<universe>)
            tv()   # short version of tv.show_player()
        """
        self.show_player(layout)
        return

    def __repr__(self):
        return('''iPlayer object:\n    {}\n    <Trajectory with {} frames>\n    <{}>'''.format(
               self.universe, len(self.universe.trajectory), self.player))

    def _init_Representation(self):
        """
        Init representation
        """
        self.color.set_color_scheme('rainbow')
        return

    def _save_camera_orientation(self):
        time.sleep(1)
        if self.player._camera_orientation_at_start == None:
            self.player._camera_orientation_at_start = self.player._camera_orientation
        return

    def _update_player_layout(self, layout='default'):
        """
        TODO DOCSTRING
        """
        self.player._layout = layout  # needed for show_plot() function
        self.widgets.YValue.layout = wg.Layout(display='none')
        if '+' in layout:
            self.widgets.YValue.layout = wg.Layout(display='visible')
        return

    def sync_view(self):
        """
        Alias for <iPlayer_object>.player.sync_view().
        """
        self.player.sync_view()
        return

    def show_player(self, layout='default'):
        """
        Show trajectory viewer with GUI.

        Args:
            layout (str):
                'default': default layout
                'Marie': special layout for special person

        Alternative:
            Execute show_player() method by calling the object.

        Example:
            tv = core.iPlayer(<universe>)
            tv()   # short version of tv.show_player()
        """
        if 'Marie' in layout:
            tv_and_widgets = wg.VBox([wg.HBox([self.widgets.FrameTime, self.widgets.YValue]),
                                      self.player,
                                      wg.HBox([self.widgets.Reset, self.widgets.UI, self.widgets.PlayerStep]),
                                      wg.HBox([wg.VBox([self.widgets.Components,
                                                        self.widgets._Component_0,
                                                        self.widgets._Component_1]),
                                               wg.VBox([self.widgets.Representation,
                                                        self.widgets._Cartoon,
                                                        self.widgets._BaS,
                                                        self.widgets._Surface,
                                                        self.widgets._Visibility]),
                                               wg.VBox([self.widgets.Color,
                                                        self.widgets._Color_red,
                                                        self.widgets._Color_orange,
                                                        self.widgets._Color_yellow,
                                                        self.widgets._Color_green,
                                                        self.widgets._Color_cyan,
                                                        self.widgets._Color_blue,
                                                        self.widgets._Color_purple,
                                                        self.widgets._Color_magenta]),
                                               wg.VBox([self.widgets.Color_by_RES,
                                                        self.widgets._Color_black,
                                                        self.widgets._Color_grey_80,
                                                        self.widgets._Color_grey_60,
                                                        self.widgets._Color_grey_40,
                                                        self.widgets._Color_grey_20,
                                                        self.widgets._Color_white,
                                                        self.widgets._Color_baw,
                                                        self.widgets._Color_rainbow])
                                               ])
                                      ])
        else:
            tv_and_widgets = wg.VBox([self.player,
                                      wg.HBox([self.widgets.FrameTime, self.widgets.YValue, self.widgets.Reset, self.widgets.UI, self.widgets.PlayerStep]),
                                      wg.HBox([wg.VBox([self.widgets.Components,
                                                        self.widgets._Component_0,
                                                        self.widgets._Component_1]),
                                               wg.VBox([self.widgets.Representation,
                                                        self.widgets._Cartoon,
                                                        self.widgets._BaS,
                                                        self.widgets._Surface,
                                                        self.widgets._Visibility]),
                                               wg.VBox([self.widgets.Color,
                                                        self.widgets._Color_red,
                                                        self.widgets._Color_orange,
                                                        self.widgets._Color_yellow,
                                                        self.widgets._Color_green,
                                                        self.widgets._Color_cyan,
                                                        self.widgets._Color_blue,
                                                        self.widgets._Color_purple,
                                                        self.widgets._Color_magenta]),
                                               wg.VBox([self.widgets.Color_by_RES,
                                                        self.widgets._Color_black,
                                                        self.widgets._Color_grey_80,
                                                        self.widgets._Color_grey_60,
                                                        self.widgets._Color_grey_40,
                                                        self.widgets._Color_grey_20,
                                                        self.widgets._Color_white,
                                                        self.widgets._Color_baw,
                                                        self.widgets._Color_rainbow])
                                               ])
                                      ])
        self._update_player_layout(layout)
        display(tv_and_widgets)
        t = threading.Thread(target=self._save_camera_orientation)
        t.start()
        return


class iPlot(iPlayer):
    def __init__(self, universe=None, xdata=None, ydata=None, xlabel='X', ylabel='Y', title='',
                 tu='ps', figsize=(8, 4.5), layout='default'):
        """
        Init iPlot.
        """
        super().__init__(universe)
        self.widgets.toggle_UI(False)
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)

        # figure properties
        self.fig = lambda: None  # create function object
        self.fig.xlabel = xlabel
        self.fig.ylabel = ylabel
        self.fig.title = title
        self.fig.tu = tu
        self.fig.figsize = figsize

        # Marie stuff start (interactive widgets)
        self._update_player_layout(layout)

        def update_YValue_description(args):
            """
            Update <iPlayer_object>.widgets.YValue.description
            """
            frame = args['new']
            # if len(ydata) == len(frames)
            if frame < len(self.ydata):
                yvalue = self.ydata[frame]
            else:
                yvalue = self.ydata[-1]
            self.widgets.YValue.description = '{}: {}'.format(self.fig.ylabel.split(" ")[0], round(yvalue, 5))
        self.player.observe(update_YValue_description, 'frame')

        def find_Min(a):
            """
            Find and jump to global minimum in xy-Plot
            """
            Xmin = self.xdata[np.argmin(self.ydata)]
            self.player.frame = int(round(Xmin/self.universe.trajectory.dt))
        self.widgets.YValue.on_click(find_Min)
        # Marie stuff end (interactive widgets)
        return

    def __call__(self, xdata=None, ydata=None, xlabel='X', ylabel='Y', title='',
                 tu='ps', figsize=(8, 4.5), layout='default'):
        """
        Show trajectory viewer with GUI and interactive matplotlib plot.
        Interactive red bar can be moved by pressing any key.

        Args:
            xdata (array)
            ydata (array)
            xlabel (str)
            ylabel (str)
            title (str)
            tu (str):
                time unit of plot. Either 'ps', 'ns' or 'frame'.
                If 'ns' is selected, time stamps of MDAnalysis universe are converted from ps to ns.
                Important to make the interactive red bar work properly.
            figsize (tuple)
            layout (str):
                'default': default layout
                'every other str': non-default layout

        Alternative:
            Execute show_plot() method by calling the object.

        Example:
            Q = core.iPlot(<universe>)
            Q()   # short version of Q.show_plot()
        """
        # special case: make the call Q('kw-layout') work as Q(layout='kw-layout')
        if type(xdata) is str:
            layout = xdata

        self._update_player_layout(layout)
        self._update_fig_properties(xlabel, ylabel, title, tu, figsize)
        self.show_player(layout)
        if xdata != None and ydata != None:
            self.show_plot(xdata, ydata, self.fig.xlabel, self.fig.ylabel, self.fig.title, self.fig.tu, self.fig.figsize)
        else:
            self.show_plot(self.xdata, self.ydata, self.fig.xlabel, self.fig.ylabel, self.fig.title, self.fig.tu, self.fig.figsize)
        return

    def __repr__(self):
        return('''iPlot object:\n    {}\n    <Trajectory with {} frames>\n    <{}>\n    <{} data points>'''.format(
               self.universe, len(self.universe.trajectory), self.player, len(self.xdata)))

    def _update_fig_properties(self, xlabel='X', ylabel='Y', title='', tu='ps', figsize=(8, 4.5), smartlabel=True):
        """
        TODO DOCSTRING
        """
        default = {'xlabel': 'X',
                   'ylabel': 'Y',
                   'title': '',
                   'tu': 'ps',
                   'figsize': (8, 4.5)}
        if xlabel != default['xlabel']:
            self.fig.xlabel = xlabel
        if ylabel != default['ylabel']:
            self.fig.ylabel = ylabel
        if title != default['title']:
            self.fig.title = title
        if tu != default['tu']:
            self.fig.tu = tu
        if figsize != default['figsize']:
            self.fig.figsize = figsize

        # extra: smartlabel (change tu and label)
        if smartlabel == True and self.fig.xlabel == default['xlabel']:
            if self.fig.tu == 'frame':
                self.fig.xlabel = 'Frame'
            elif self.fig.tu == 'ps':
                self.fig.xlabel = 'Time (ps)'
            elif self.fig.tu == 'ns':
                self.fig.xlabel = 'Time (ns)'
        if smartlabel == True and 'frame' in self.fig.xlabel.lower():
            self.fig.tu = 'frame'
        if smartlabel == True and self.fig.ylabel == default['ylabel']:
            self.fig.ylabel = 'RMSD'
        return

    def show_plot(self, xdata=None, ydata=None, xlabel='X', ylabel='Y', title='',
                  tu='ps', figsize=(8, 4.5)):
        """
        Show trajectory viewer with GUI and interactive matplotlib plot.
        Interactive red bar can be moved by pressing any key.

        Args:
            xdata (array)
            ydata (array)
            xlabel (str)
            ylabel (str)
            title (str)
            tu (str):
                time unit of plot. Either 'ps', 'ns' or 'frame'.
                If 'ns' is selected, time stamps of MDAnalysis universe are converted from ps to ns.
                Important to make the interactive red bar work properly.
            figsize (tuple)

        Alternative:
            Execute show_plot() method by calling the object.

        Example:
            Q = core.iPlot(<universe>)
            Q()   # short version of Q.show_plot()
        """
        self._update_player_layout(self.player._layout)
        self._update_fig_properties(xlabel, ylabel, title, tu, figsize)

        # assign values
        if np.all(xdata) != None and np.all(ydata) != None:
            self.xdata = np.array(xdata)
            self.ydata = np.array(ydata)

        # assign pseudo values for testing (if no values assigned yet)
        if np.all(self.xdata) == None and np.all(self.ydata) == None:
            print("No data specified. Using pseudo values for figure.\
                 \nJump to a frame/move the interactive red bar by holding any key and pressing LMB.")
            if self.fig.tu == 'frame':
                self.xdata = np.linspace(self.universe.trajectory[0].frame,
                                         self.universe.trajectory[-1].frame,
                                         100)
            else:
                self.xdata = np.linspace(self.universe.trajectory[0].time,
                                         self.universe.trajectory[-1].time,
                                         100)
            self.ydata = np.random.uniform(low=0.0, high=5.0, size=(100,))

        # plot
        self.fig.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.fig.add_subplot(111)
        if self.fig.tu == 'ps':
            # self.ax.plot(self.xdata, self.ydata, 'b.', ms=1, lw=None)
            self.ax.plot(self.xdata, self.ydata, color=cp[0], alpha=0.85, lw=2)
            self.ax.fill_between(self.xdata, self.ydata, color=cp[0], alpha=0.15)
            try:
                plt.axvline(x=self.xdata[0], color="red", lw=2)  # redline
                plt.xlim(self.xdata[0], self.xdata[-1])
            except TypeError or IndexError:
                pass
        if self.fig.tu == 'ns':
            self._xdata_ns = 0.001 * self.xdata
            # self.ax.plot(self._xdata_ns, self.ydata, 'b.', ms=1, lw=None)
            self.ax.plot(self._xdata_ns, self.ydata, color=cp[0], alpha=0.85, lw=2)
            self.ax.fill_between(self._xdata_ns, self.ydata, color=cp[0], alpha=0.15)
            try:
                plt.axvline(x=self._xdata_ns[0], color="red", lw=2)  # redline
                plt.xlim(self._xdata_ns[0], self._xdata_ns[-1])
            except TypeError or IndexError:
                pass
        if self.fig.tu == 'frame':
            self._xdata_frame = np.arange(0, len(self.xdata))
            # self.ax.plot(self._xdata_frame, self.ydata, 'b.', ms=1, lw=None)
            self.ax.plot(self._xdata_frame, self.ydata, color=cp[0], alpha=0.85, lw=2)
            self.ax.fill_between(self._xdata_frame, self.ydata, color=cp[0], alpha=0.15)
            try:
                plt.axvline(x=self._xdata_frame[0], color="red", lw=2)  # redline
                plt.xlim(self._xdata_frame[0], self._xdata_frame[-1])
            except TypeError or IndexError:
                pass

        plt.xlabel(self.fig.xlabel)
        plt.ylabel(self.fig.ylabel)
        if self.fig.title != '':
            plt.title(self.fig.title, fontweight="bold")
        plt.ylim(0,)
        sns.despine(offset=0)
        plt.tight_layout()

        # make fig interactive
        self._event = lambda: None
        self._event.key_is_held = False

        def hold_key(event):
            self._event.key_is_held = True
            return
        self.fig.fig.canvas.mpl_connect('key_press_event', hold_key)

        def release_key(event):
            self._event.key_is_held = False
            return
        self.fig.fig.canvas.mpl_connect('key_release_event', release_key)

        def onclick_goto_FRAME(event):
            """
            Click behavior for tu='frame'.
            """
            if self._event.key_is_held and self.fig.tu == 'frame':
                self.player.frame = int(round(event.xdata))
            return
        self.fig.fig.canvas.mpl_connect('button_press_event', onclick_goto_FRAME)

        def onclick_goto_TIME(event):
            """
            Click behavior for tu='ps' and tu='ns'.
            """
            if self._event.key_is_held and self.fig.tu == 'ps':
                self.player.frame = int(round(event.xdata / self.universe.trajectory.dt))
            elif self._event.key_is_held and self.fig.tu == 'ns':
                self.player.frame = int(round(1000 * event.xdata / self.universe.trajectory.dt))
            return
        self.fig.fig.canvas.mpl_connect('button_press_event', onclick_goto_TIME)

        def draw_redbar(args):
            """
            Draw red bar in interactive matplotlib based on current frame/time of trajectory viewer.
            """
            frame = args['new']
            del self.ax.lines[-1]

            if self.fig.tu == 'frame':
                self.ax.axvline(x=frame, color="red", lw=2)
            elif self.fig.tu == 'ps':
                time = frame * self.universe.trajectory.dt
                self.ax.axvline(x=time, color="red", lw=2)
            elif self.fig.tu == 'ns':
                time = 0.001 * frame * self.universe.trajectory.dt
                self.ax.axvline(x=time, color="red", lw=2)
            return
        self.player.observe(draw_redbar, 'frame')
        return
