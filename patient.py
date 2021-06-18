# patient object class and construtor

class Patient:
    def __init__(self, id, hx, sx, vs):
        self.id = id
        self.hx = hx
        self.sx = sx
        self.vs = vs


    def updateDatabase(self):
        #format data to DB
        # update approriate DB