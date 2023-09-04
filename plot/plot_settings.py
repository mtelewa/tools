#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import label_lines

class Initialize:
    """
    Initialize the plot
    """
    def __init__(self, configfile):
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)

    def create_fig(self):
        """
        Parameters:
        -----------
        Creates the canvas with the figure settings based on the number of subfigures
        specified from 'ncols' and 'nrows' from in the Yaml config file

        Returns:
        --------
        Fig: matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
        axes_array: numpy.ndarray
        """
        # A figure with nRows and nColumns
        nrows = self.config['nrows']
        ncols = self.config['ncols']

        # One plot per figure
        if nrows==1 and ncols==1:
            fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)#, figsize=(6.4, 3.2))#, figsize=(7.8,6.8)) #,figsize=(4.6, 4.1))
            axes_array = np.array(ax).reshape(-1)

        # Multiple subplots
        if nrows > 1 or ncols > 1:
            if nrows>1:
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, figsize=(6.4, 5.2))#figsize=(7,7))
                fig.subplots_adjust(hspace=0.05)         # Adjust space between axes
            if ncols>1:
                fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharey=True, figsize=(8,7))
                fig.subplots_adjust(wspace=0.05)
            axes_array = ax.reshape(-1)

            if self.config['broken_axis'] is not None:
                # Hide the bottom spines and ticks of all the axes except the last (bottom) one
                if len(axes_array)>2:
                    for ax in axes_array[:-2]: # [:-1] # TODO:
                        ax.spines.bottom.set_visible(False)
                        ax.tick_params(labeltop=False, bottom=False)  # don't put tick labels at the bottom
                    # Hide the top spines and ticks of all the axes except the first (top) one
                    for ax in axes_array[1:-1]: #[1:]
                        ax.spines.top.set_visible(False)
                        ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top
                else:
                    for ax in axes_array[:-1]:
                        ax.spines.bottom.set_visible(False)
                        ax.tick_params(labeltop=False, bottom=False)  # don't put tick labels at the bottom
                    # Hide the top spines and ticks of all the axes except the first (top) one
                    for ax in axes_array[1:]:
                        ax.spines.top.set_visible(False)
                        ax.tick_params(labeltop=False, top=False)  # don't put tick labels at the top

        # 3D plots
        if self.config['3d']:
            fig, ax = plt.figure(dpi=1200), plt.axes(projection='3d')

        # Heat map
        try:
            if self.config['heat']:
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(6.4,5.3))
        except KeyError:
            pass

        # Animation
        try:
            if self.config['anim']:
                fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
        except KeyError:
            pass

        return {'fig': fig, 'ax': ax, 'axes_array': axes_array}


class Modify:
    """
    Reads the plot settings from the Yaml file and modifies the created plot
    """

    def __init__(self, xdata, fig, axes_array, configfile):

        self.xdata = xdata
        self.fig = fig
        # Read the yaml file
        with open(configfile, 'r') as f:
            self.config = yaml.safe_load(f)

        # Make a list that contains all the lines in all the axes starting from the last axes
        lines = []
        for ax in axes_array:
            lines.append(list(ax.get_lines()))

        lines = [item for sublist in lines for item in sublist]
        print(f'Lines on the figure: {len(lines)}')

        # Modifies the linestyle, marker, color, label and opacity
        for i, line in enumerate(lines):
            if self.config[f'lstyle_{i}']=='pop':
                line.set_linestyle(' ')
                line.set_marker(' ')
                line.set_label(None)
            else:
                line.set_linestyle(self.config[f'lstyle_{i}'])
                line.set_marker(self.config[f'mstyle_{i}'])
                if self.config[f'color_{i}']: # If color is pre-defined
                    line.set_color(self.config[f'color_{i}'])
                line.set_label(self.config[f'label_{i}'])
                line.set_alpha(self.config[f'alpha_{i}'])
            print(f'Line {i} label: {line.get_label()}')

        # Additional modifiers ---------------------
        # ------------------------------------------
        # Set the axes limits
        for i, ax in enumerate(axes_array):
            if self.config['xlabel_0'] != None or self.config['ylabel_0'] != None:
                axes_array[i].set_xlabel(self.config[f'xlabel_{i}'])
                axes_array[i].set_ylabel(self.config[f'ylabel_{i}'])
            if self.config[f'xlo_{i}'] is not None: ax.set_xlim(left=self.config[f'xlo_{i}'])
            if self.config[f'xhi_{i}'] is not None: ax.set_xlim(right=self.config[f'xhi_{i}']*np.max(self.xdata))
            if self.config[f'ylo_{i}'] is not None: ax.set_ylim(bottom=self.config[f'ylo_{i}'])
            if self.config[f'yhi_{i}'] is not None: ax.set_ylim(top=self.config[f'yhi_{i}'])
        if self.config['legend_elements'] is not None: self.add_legend(axes_array[self.config['legend_on_ax']])
        if self.config['label_x-pos'] is not None: self.label_inline(lines)
        if self.config['label_subplot'] is not None: self.label_subplot(axes_array)
        if self.config['vertical_line_pos_1'] is not None: self.plot_vlines(axes_array)
        if self.config['broken_axis'] is not None:
            if self.config['shared_label'] is None:
                self.plot_broken(axes_array)
            else:
                self.plot_broken(axes_array, shared_label=1)
        if self.config['set_ax_height'] is not None: self.set_ax_height(axes_array)
        # if self.config['plot_inset'] is not None: self.plot_inset(axes_array)
        try:
            if self.config['hline_pos'] is not None: self.plot_hlines(axes_array)
        except KeyError:
            pass

    def add_legend(self, axis):
        """
        Modifies the legend by adding user-specified elements
        """
        handles, labels = axis.get_legend_handles_labels()
        #Additional elements
        # TODO: Generalize
        legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quadratic fit'),
                           Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Quartic fit')]
        #                    Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Lin. extrapolation')]
        # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Fluid'),
        #                   Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Wall')]
        # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='$C\dot{\gamma}^{n}$')]
        # legend_elements = [Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='^', markersize=8, label='Fixed Force'),
        #                    Line2D([0], [0], color='k', lw=2.5, ls=' ', marker='v', markersize=8, label='Fixed Current'),
        #                    Line2D([0], [0], color='k', lw=2.5, ls='-', marker=' ', label='Quadratic fit')]
        # #                    Line2D([0], [0], color='k', lw=2.5, ls='--', marker=' ', label='Wall $\sigma_{zz}$')]

        handles.extend(legend_elements)

        # Extend legend items from handles to those specified in elements
        if self.config['legend_elements']=='h+e':
            axis.legend(handles=handles, frameon=False)
        # Import legend items from the elements specified
        elif self.config['legend_elements']=='e':
            if self.config['legend_loc'] == 1:
                axis.legend(handles=legend_elements, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 0.4), columnspacing=1, ncol=1)
                # axis.legend(handles=legend_elements, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.5), columnspacing=1, ncol=3)#loc=(0.,0.45))
            if isinstance(self.config['legend_loc'], str):
                axis.legend(handles=legend_elements, frameon=False, loc=self.config['legend_loc'])
            if self.config['legend_loc'] is None:
                axis.legend(handles=legend_elements, frameon=False)
        # Import legend items from the handles
        elif self.config['legend_elements']=='h':
            if self.config['legend_loc'] == 1:
                axis.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4)
                # axis.legend(frameon=False, loc='upper right')
            else:
                axis.legend(frameon=False)#, markerscale=2)


    def label_inline(self, lines):
        """
        Sets labels on the plotted lines
        """

        for i, line in enumerate(lines):
            # Ignore the lines with no labels and/or data which is not plotted
            if line.get_linestyle() == 'pop' or line.get_label() == ' ' or line.get_label() is None: #.startswith('_'):
                pass
            else:
                xpos = self.config[f'label_x-pos_{i}']
                rot = self.config[f'rotation_of_label_{i}']
                y_offset = self.config[f'Y-offset_for_label_{i}']
                label_lines.label_line(line, xpos, yoffset= y_offset, \
                         label=line.get_label(), rotation= rot)


    def label_subplot(self, axes):
        """
        Sets labels on the subplots
        """
        sublabels=('(a)', '(b)', '(c)', '(d)')

        if self.config['broken_axis'] is None:
            for i, ax in enumerate(axes):
                axes[i].text(-0.1, 1.1, sublabels[i], transform=ax.transAxes,
                        fontweight='bold', va='top', ha='right')
        else: # TODO: Generalize later
            last_axis = len(axes)-1
            axes[0].text(-0.1, 1.1, sublabels[0], transform=axes[0].transAxes,
                        fontweight='bold', va='top', ha='right')
            axes[last_axis].text(-0.1, 1.1, sublabels[1], transform=axes[last_axis].transAxes,
                        fontweight='bold', va='top', ha='right')


    def plot_vlines(self, axes):
        """
        Plots vertical lines if the position was given in the config file
        """
        # Draw vlines only if the position was given in the  yaml file
        pos1 = self.config['vertical_line_pos_1']
        pos2 = self.config['vertical_line_pos_2']

        for ax in range(len(axes)):
            axes[ax].axvline(x= pos1*np.max(self.xdata), color='k', marker=' ', linestyle='dotted', lw=1.5)
            axes[ax].axvline(x= pos2*np.max(self.xdata), color='k', marker=' ', linestyle='dotted', lw=1.5)

    def plot_hlines(self, axes):
        """
        Plots horizontal lines if the position was given in the config file
        """
        # Draw vlines only if the position was given in the  yaml file
        pos1 = self.config['hline_pos']

        for ax in range(len(axes)):
            axes[ax].axhline(y= pos1, color='k', marker=' ', linestyle='dashdot', lw=1.5)


    def plot_broken(self, axes, shared_label=None):
        """
        Plots broken axes for a more convenient display of a large data range
        """
        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                      linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        # Draw the dashes
        axes[0].plot([0, 1], [0, 0], transform=axes[0].transAxes, **kwargs)
        axes[1].plot([0, 1], [1, 1], transform=axes[1].transAxes, **kwargs)

        if len(axes)>3:
            axes[1].plot([0, 1], [0, 0], transform=axes[1].transAxes, **kwargs)
            axes[2].plot([0, 1], [1, 1], transform=axes[2].transAxes, **kwargs)

        # Remove all axes label
        for ax in axes:
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)

        if shared_label:
            self.fig.supylabel(axes[0].get_ylabel())
            self.fig.text(0.5, 0.005, axes[-1].get_xlabel(), ha='center')
        else:
            #Set the common labels # TODO : generalize the y-axis labels and their positions
            self.fig.text(0.5, 0.04, axes[-1].get_xlabel(), ha='center')
            self.fig.text(0.04, 0.70, axes[0].get_ylabel(), va='center', rotation='vertical')
            self.fig.text(0.04, 0.30, axes[-1].get_ylabel(), va='center', rotation='vertical')


    def set_ax_height(self, axes):
        #gs = GridSpec(len(pt.axes_array), 1, height_ratios=[1,1,1,2])
        list = np.ones(len(axes))
        list[-1] = 2
        gs = GridSpec(len(axes), 1, height_ratios=list)
        for i, ax in enumerate(axes):
            ax.set_position(gs[i].get_position(self.fig))
