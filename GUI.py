import tkinter as tk
import numpy as np
import math
import time
import os

class GUI(object):
    def __init__(self,environment_details,arena_select,car_details,graphs,trace=False):
        self.env = environment_details
        self.w,self.h = self.env['display_resolution']
        self.graphs = graphs
        self.runs = ['Learn','Test']
        self.COLORS = ['RoyalBlue2','SlateGray2','dark khaki','DarkOrange1','turquoise1','indian red','MediumPurple1','lemon chiffon','PaleVioletRed1','gold','PaleGreen1','salmon2'] if environment_details['color_coding']==True else ['blue']*12
        self.construct_window()
        self.title_label('Reinforcement Learning Car Simulation')
        self.cars = [dict(car) for car in car_details]
        self.trace = trace
        self.trace_history_limit = self.env['trace_history_limit']
        self.init_env(arena_select.split(','))
        self.init_car()
        self.init_graph()
        self.approximator = lambda x: str(round(x, -int(math.floor(math.log10(abs(x))))+3)) if x!=0 else '0'
        self.refresh()
        # Mouse and Keypress event
        self.mouse_click_loaction = [None,None]
        self.key_pressed = None
        os.system('xset r on') # Repeat interrupt when kept pressed
        # GUI constants
        self.use_arrow = True

    def construct_window(self):
        # Level 1: Main window and frame
        self.window = tk.Tk()
        self.window.resizable(0,0)
        self.window_frame = tk.Frame(self.window,padx=10,pady=10)
        self.window_frame.pack()
        # Level 2: Segments for display and options
        proportion = 0.7
        self.display = tk.Canvas(self.window_frame, width=int(proportion*self.w), height=self.h)
        self.display.bind("<Button-1>", self.mouse_click) #Misc
        self.display.bind("<Key>", self.keypress)
        self.display.focus_set()
        self.display.pack(side=tk.LEFT)
        self.display.config(background='gray90')
        self.options = tk.Frame(self.window_frame,padx=25,pady=25)
        self.options.pack(side=tk.RIGHT,fill=tk.Y)
        self.options.config(background='white smoke')
        # Level 3: Graph and debug info inside options. Display has no other children
        self.graph = tk.Canvas(self.options,background='white',width=int((1-proportion)*self.w), height=int(self.h/2))
        self.graph.pack(side=tk.BOTTOM)
        self.run_select = tk.StringVar()
        self.run_select_button_holder = tk.Frame(self.options)
        self.run_select_button_holder.pack(side=tk.TOP,anchor=tk.CENTER)
        self.debug_info = tk.StringVar()
        tk.Label(self.options,textvariable=self.debug_info,justify=tk.LEFT,background='white smoke',pady=10,font=('System')).pack(side=tk.TOP)
        self.graph_select = tk.StringVar()
        self.graph_select.set(self.graphs[0])
        self.graph_select_button_holder = tk.Frame(self.options)
        self.graph_select_button_holder.pack(side=tk.BOTTOM,anchor=tk.CENTER)
        # Level 4: Configure buttons
        for r in self.runs:
            tk.Radiobutton(self.run_select_button_holder, text=r, variable=self.run_select, value=r, indicatoron=0,font=('System')).pack(side=tk.LEFT,anchor=tk.CENTER)
        for g in self.graphs:
            tk.Radiobutton(self.graph_select_button_holder, text=g, variable=self.graph_select, value=g, indicatoron=0,font=('System')).pack(side=tk.LEFT,anchor=tk.CENTER)

    def set_display_range(self,xmin,xmax,ymin,ymax):
        x_range,y_range = xmax-xmin+2,ymax-ymin+2
        self.display_w, self.display_h = int(self.display.config()['width'][-1]),int(self.display.config()['height'][-1])
        self.scale_factor = min(self.display_w/x_range, self.display_h/y_range)
        self.center_offset = ((1-xmin)*self.scale_factor,(1-ymin)*self.scale_factor)

    def scale_and_offset_center(self,list_of_points):
        tr_pts = []
        for pt in list_of_points:
            tr_pts.append(int((pt[0]*self.scale_factor)+self.center_offset[0]))
            tr_pts.append(self.display_h-int((pt[1]*self.scale_factor)+self.center_offset[1]))
        return tr_pts

    def inverse_scale_and_offset_center(self,list_of_points):
        tr_pts = []
        for pt in list_of_points:
            tr_pts.append( [ round((float(pt[0])-self.center_offset[0])/self.scale_factor,2) , round((self.display_h-float(pt[1])-self.center_offset[1])/self.scale_factor,2)] )
        return tr_pts

    def rotation_matrix(self,theta):
        ct = math.cos(theta)
        st = math.sin(theta)
        R = np.array([[ct,-st],[st,ct]])
        return R

    def init_destination(self,reinit,*agents):
        radius = self.env['dest_radius']
        if reinit==False:
            self.destination = []
            for idx,agent in enumerate(agents):
                center = [agent.destination.x,agent.destination.y]
                dest_pts = [(center[0]-radius,center[1]-radius),(center[0]+radius,center[1]+radius)]
                dest_pts = self.scale_and_offset_center(dest_pts)
                dest_id = self.display.create_oval(dest_pts,outline=self.COLORS[idx%12],width=10)
                self.destination.append(center+[dest_id])
        else:
            for idx,agent in enumerate(agents):
                center = [agent.destination.x,agent.destination.y]
                if self.destination[idx][0]==center[0] and self.destination[idx][1]==center[1]:
                    continue
                dest_pts = [(center[0]-radius,center[1]-radius),(center[0]+radius,center[1]+radius)]
                dest_pts = self.scale_and_offset_center(dest_pts)
                dest_id = self.destination[idx][2]
                self.display.coords(dest_id, *dest_pts)
                self.destination[idx] = center+[dest_id]

    def init_env(self,select):
        x_list,y_list = [],[]
        for arena in select:
            if not arena in self.env['Arena']:
                raise Exception('Seleted arena '+arena+' not defined in configuration')
            for pt in self.env['Arena'][arena]['points']:
                x_list.append(pt[0])
                y_list.append(pt[1])
        xmin,xmax,ymin,ymax = min(x_list),max(x_list),min(y_list),max(y_list)
        self.set_display_range(xmin,xmax,ymin,ymax)
        for arena in select:
            arena_coords = self.scale_and_offset_center(self.env['Arena'][arena]['points'])
            self.arena_id = self.display.create_polygon(arena_coords,fill='white',outline='black')
            if 'obstacles' in self.env['Arena'][arena]:
                for obs in self.env['Arena'][arena]['obstacles']:
                    obs_coords = self.scale_and_offset_center(obs)
                    self.display.create_polygon(obs_coords,fill='gray30',outline='black')

    def init_car(self):
        for i in range(len(self.cars)):
            # Trace
            self.trace_mod(i,self.cars[i]['state'][0:2],init=True)
        for i in range(len(self.cars)):
            # Car
            R = self.rotation_matrix(self.cars[i]['state'][2])
            w = self.cars[i]['W']
            l = self.cars[i]['L']
            points = np.array([ [0,-w], [l,-w], [l,w], [0,w] ]).T
            points = np.dot(R,points)
            points[0,:] += self.cars[i]['state'][0]
            points[1,:] += self.cars[i]['state'][1]
            self.cars[i]['gui_body_id'] = self.display.create_polygon(self.scale_and_offset_center(points.T),fill=self.COLORS[i%12],outline='black')
            # Sensor
            for j in range(len(self.cars[i]['sensors'])):
                sa,sr = self.cars[i]['sensors'][j]['angle'],self.cars[i]['sensors'][j]['range']
                R = self.rotation_matrix(self.cars[i]['state'][2]+sa)
                points = np.array([ [0,0],[sr,0] ]).T
                points = np.dot(R,points)
                points[0,:] += self.cars[i]['state'][0]
                points[1,:] += self.cars[i]['state'][1]
                self.cars[i]['sensors'][j]['gui_sensor_id'] = self.display.create_line(self.scale_and_offset_center(points.T),fill='red')

    def init_graph(self):
        self.graph_w,self.graph_h = int(self.graph.config()['width'][-1]),int(self.graph.config()['height'][-1])
        self.plots = [self.graph.create_line([(0,0),(1,1)],width=2.0),
                        self.graph.create_text(0,self.graph_h,anchor=tk.SW,text=str((0,0))),
                        self.graph.create_text(0,0,anchor=tk.NW,text=str(1)),
                        self.graph.create_text(self.graph_w,self.graph_h,anchor=tk.SE,text=str(1))]

    def trace_mod(self,car_id,pt,init=False,force_end_line=False):
        if init==True:
            self.cars[car_id]['trace_history'] = [-1]*self.trace_history_limit
            self.cars[car_id]['trace_history_index'] = 0
            self.cars[car_id]['trace_history_buffer'] = []
        self.cars[car_id]['trace_history_buffer'].append(pt)
        if len(self.cars[car_id]['trace_history_buffer'])>100 or force_end_line==True:
            self.display.delete(self.cars[car_id]['trace_history'][self.cars[car_id]['trace_history_index']])
            if len(self.cars[car_id]['trace_history_buffer'])>2:
                if self.use_arrow==True:
                    self.cars[car_id]['trace_history'][self.cars[car_id]['trace_history_index']] = self.display.create_line(self.scale_and_offset_center(self.cars[car_id]['trace_history_buffer']),fill=self.COLORS[car_id%12],width=1.5,arrow=tk.LAST,arrowshape=(8,10,6))
                else:
                    self.cars[car_id]['trace_history'][self.cars[car_id]['trace_history_index']] = self.display.create_line(self.scale_and_offset_center(self.cars[car_id]['trace_history_buffer']),fill=self.COLORS[car_id%12],width=1.0)
                self.cars[car_id]['trace_history_index'] += 1
                if self.cars[car_id]['trace_history_index'] >= self.trace_history_limit: self.cars[car_id]['trace_history_index'] = 0
            self.cars[car_id]['trace_history_buffer'] = []

    def update(self,car_id,state,draw_car=True,force_end_line=False):
        if self.trace==True:
            self.trace_mod(car_id,state[0:2],force_end_line=force_end_line)
        if draw_car==True:
            R = self.rotation_matrix(state[2])
            w = self.cars[car_id]['W']
            l = self.cars[car_id]['L']
            points = np.array([ [0,-w], [l,-w], [l,w], [0,w] ]).T
            points = np.dot(R,points)
            points[0,:] += state[0]
            points[1,:] += state[1]
            translated_points = self.scale_and_offset_center(points.T)
            self.display.coords(self.cars[car_id]['gui_body_id'], *translated_points)
            self.display.tag_raise(self.cars[car_id]['gui_body_id'])
            for j in range(len(self.cars[car_id]['sensors'])):
                sa,sr = self.cars[car_id]['sensors'][j]['angle'],self.cars[car_id]['sensors'][j]['range']
                R = self.rotation_matrix(state[2]+sa)
                points = np.array([ [0,0],[sr,0] ]).T
                points = np.dot(R,points)
                points[0,:] += state[0]
                points[1,:] += state[1]
                translated_points = self.scale_and_offset_center(points.T)
                self.display.coords(self.cars[car_id]['sensors'][j]['gui_sensor_id'],*translated_points)
                self.display.tag_raise(self.cars[car_id]['sensors'][j]['gui_sensor_id'])

    def update_graph(self,X,Y,plt_idx=None,color='black',thickness=1.0,style=None):
        if plt_idx != self.graph_select.get(): return
        if len(X)<2: return
        minX,minY,maxX,maxY = min(X),min(Y),max(X),max(Y)
        w,h = maxX-minX, maxY-minY
        yscale = minY if h==0 else float(self.graph_h)/h
        xscale = minX if w==0 else float(self.graph_w)/w
        scaled_pts = []
        for (x,y) in zip(X,Y):
            scaled_pts.append(int(xscale*(x-minX)))
            scaled_pts.append(self.graph_h - int(yscale*(y-minY)))
        self.graph.coords(self.plots[0],*scaled_pts)
        self.graph.itemconfig(self.plots[1],text=self.approximator(minX)+','+self.approximator(minY))
        self.graph.itemconfig(self.plots[2],text=self.approximator(maxY))
        self.graph.itemconfig(self.plots[3],text=self.approximator(maxX))

    def update_debug_info(self,info):
        self.debug_info.set(info)

    def refresh(self):
        self.window.update()
        self.window.update_idletasks()
        time.sleep(self.env['display_dk'])

    def enable_trace(self,remove_traces=False):
        self.trace = True
        if remove_traces==True: self.remove_traces()

    def disable_trace(self,remove_traces=False):
        self.trace = False
        if remove_traces==True: self.remove_traces()

    def title_label(self,label):
        self.window.title(label)

    def get_run_select(self):
        return self.run_select.get()

    def set_run_select(self,value):
        self.run_select.set(value)

    def remove_traces(self):
        for car in self.cars:
            car['trace_history_index'] = 0
            car['trace_history_buffer'] = []
            for idx in range(len(car['trace_history'])):
                self.display.delete(car['trace_history'][idx])

    def mouse_click(self,event):
        self.mouse_click_loaction = self.inverse_scale_and_offset_center([[event.x,event.y]])[0]

    def keypress(self,event):
        self.key_pressed = event.char

    def get_userinput(self):
        uk = str(self.key_pressed) if self.key_pressed is not None else None
        self.key_pressed = None
        return uk

    def create_marker(self,pos,style,l=0.1,angle=None):
        if style=='o':
            dest_pts = [(pos[0]-l,pos[1]-l),(pos[0]+l,pos[1]+l)]
            dest_pts = self.scale_and_offset_center(dest_pts)
            dest_id = self.display.create_oval(dest_pts,fill='red')
        elif style=='x':
            points = [ [-l+pos[0],-l+pos[1]], [l+pos[0],l+pos[1]] ]
            self.display.create_line(self.scale_and_offset_center(points),fill='green',width=2.5)
            points = [ [-l+pos[0],l+pos[1]], [l+pos[0],-l+pos[1]] ]
            self.display.create_line(self.scale_and_offset_center(points),fill='green',width=2.5)
        elif style=='arrow':
            R = self.rotation_matrix(angle)
            points = np.array([ [0,0], [l,0] ]).T
            points = np.dot(R,points)
            points[0,:] += pos[0]
            points[1,:] += pos[1]
            self.display.create_line(self.scale_and_offset_center(points.T),fill='red',width=4,arrow=tk.LAST,arrowshape=(8,12,5))

    def create_label(self,pos,text):
        pos = list(pos)
        pos[1] += -0.2
        pos[0] += 0.3
        self.display.create_text(self.scale_and_offset_center([pos]),text=text,justify=tk.CENTER,font=('System'),fill='green')

    def sleep(self,seconds):
        time.sleep(seconds)
