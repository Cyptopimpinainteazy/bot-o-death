import tkinter as tk
from tkinter import ttk

class ToggleSwitch(tk.Canvas):
    def __init__(self, parent, width=60, height=30, bg_color='#2E3440', fg_color='#88C0D0', **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0, **kwargs)
        self.bg_color = bg_color
        self.fg_color = fg_color
        self.state = False
        self.width = width
        self.height = height
        
        # Create the switch background
        self.switch_bg = self.create_rounded_rect(4, 4, width-4, height-4, radius=height//2-4, fill='#4C566A')
        
        # Create the toggle button
        self.toggle_button = self.create_oval(4, 4, height-4, height-4, fill='#D8DEE9')
        
        self.bind('<Button-1>', self.toggle)
        
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
        
    def toggle(self, event=None):
        self.state = not self.state
        if self.state:
            self.itemconfig(self.switch_bg, fill=self.fg_color)
            self.coords(self.toggle_button, 
                      self.width-self.height+4, 4, 
                      self.width-4, self.height-4)
        else:
            self.itemconfig(self.switch_bg, fill='#4C566A')
            self.coords(self.toggle_button, 
                      4, 4, 
                      self.height-4, self.height-4)
        
        if hasattr(self, 'command'):
            self.command()
            
    def get(self):
        return self.state
        
    def set(self, state):
        if self.state != state:
            self.toggle()
            
    def configure(self, **kwargs):
        if 'command' in kwargs:
            self.command = kwargs.pop('command')
        super().configure(**kwargs)
