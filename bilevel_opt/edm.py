"""
Wrappers for edm4hep ROOT objects.

"""

from __future__ import annotations

from math import acos, atan2, sqrt

import numpy as np


class MCParticle:
    def __init__(self, mcp):
        self.PDG                = mcp.PDG
        self.generatorStatus    = mcp.generatorStatus
        self.simulatorStatus    = mcp.simulatorStatus
        self.charge             = mcp.charge
        self.time               = mcp.time
        self.mass               = mcp.mass
        self.vx                 = mcp.vertex.x
        self.vy                 = mcp.vertex.y
        self.vz                 = mcp.vertex.z
        self.endx               = mcp.endpoint.x
        self.endy               = mcp.endpoint.y
        self.endz               = mcp.endpoint.z
        self.px                 = mcp.momentum.x
        self.py                 = mcp.momentum.y
        self.pz                 = mcp.momentum.z
        self.endpx              = mcp.momentumAtEndpoint.x
        self.endpy              = mcp.momentumAtEndpoint.y
        self.endpz              = mcp.momentumAtEndpoint.z
        self.spin               = mcp.spin
        self.colorFlow          = mcp.colorFlow
        self.energy             = sqrt(self.px**2 + self.py**2 + self.pz**2 + self.mass**2)
        self.pathlength         = sqrt((self.vx - self.endx)**2
                                     + (self.vy - self.endy)**2
                                     + (self.vz - self.endz)**2)


class MCCollection:
    def __init__(self, mcplist):
        self.particles           = np.array(mcplist)
        self.N                   = len(mcplist)
        self.PDG                 = np.array([mcp.PDG                 for mcp in self.particles])
        self.generatorStatus     = np.array([mcp.generatorStatus     for mcp in self.particles])
        self.simulatorStatus     = np.array([mcp.simulatorStatus     for mcp in self.particles])
        self.charge              = np.array([mcp.charge              for mcp in self.particles])
        self.time                = np.array([mcp.time                for mcp in self.particles])
        self.mass                = np.array([mcp.mass                for mcp in self.particles])
        self.vx                  = np.array([mcp.vx                  for mcp in self.particles])
        self.vy                  = np.array([mcp.vy                  for mcp in self.particles])
        self.vz                  = np.array([mcp.vz                  for mcp in self.particles])
        self.endx                = np.array([mcp.endx                for mcp in self.particles])
        self.endy                = np.array([mcp.endy                for mcp in self.particles])
        self.endz                = np.array([mcp.endz                for mcp in self.particles])
        self.px                  = np.array([mcp.px                  for mcp in self.particles])
        self.py                  = np.array([mcp.py                  for mcp in self.particles])
        self.pz                  = np.array([mcp.pz                  for mcp in self.particles])
        self.endpx               = np.array([mcp.endpx               for mcp in self.particles])
        self.endpy               = np.array([mcp.endpy               for mcp in self.particles])
        self.endpz               = np.array([mcp.endpz               for mcp in self.particles])
        self.spin                = np.array([mcp.spin                for mcp in self.particles])
        self.colorFlow           = np.array([mcp.colorFlow           for mcp in self.particles])
        self.energy              = np.array([mcp.energy              for mcp in self.particles])

    def __iter__(self):
        for mcp in self.particles:
            yield mcp


class MCContrib:
    def __init__(self, mcp):
        self.PDG                 = mcp.PDG
        self.energy              = mcp.energy
        self.time                = mcp.time
        self.x                   = mcp.stepPosition.x
        self.y                   = mcp.stepPosition.y
        self.z                   = mcp.stepPosition.z


class MCContribCollection:
    def __init__(self, mcplist):
        self.contribs            = np.array(mcplist)
        self.N                   = len(mcplist)
        self.PDG                 = np.array([mcp.PDG                 for mcp in self.contribs])
        self.energy              = np.array([mcp.energy              for mcp in self.contribs])
        self.time                = np.array([mcp.time                for mcp in self.contribs])
        self.x                   = np.array([mcp.x                   for mcp in self.contribs])
        self.y                   = np.array([mcp.y                   for mcp in self.contribs])
        self.z                   = np.array([mcp.z                   for mcp in self.contribs])

    def __iter__(self):
        for mcp in self.contribs:
            yield mcp


class SimTrackerHit:
    def __init__(self, hit):
        self.cellID           = hit.cellID
        self.E                = hit.eDep
        self.time             = hit.time
        self.pathlength       = hit.pathLength
        self.quality          = hit.quality
        self.x                = hit.position.x
        self.y                = hit.position.y
        self.z                = hit.position.z
        self.px               = hit.momentum.x
        self.py               = hit.momentum.y
        self.pz               = hit.momentum.z
        self.r                = sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta            = acos(self.z / self.r) if self.r != 0 else 0
        self.phi              = atan2(self.y, self.x)


class SimCalorimeterHit:
    def __init__(self, hit):
        self.cellID           = hit.cellID
        self.E                = hit.energy
        self.x                = hit.position.x
        self.y                = hit.position.y
        self.z                = hit.position.z
        self.r                = sqrt(self.x**2 + self.y**2 + self.z**2)
        self.theta            = acos(self.z / self.r) if self.r != 0 else 0
        self.phi              = atan2(self.y, self.x)


class SimTrackerHitCollection:
    def __init__(self, rawhits):
        self.hits             = np.array(rawhits)
        self.N                = len(rawhits)
        self.cellID           = np.array([hit.cellID      for hit in self.hits])
        self.E                = np.array([hit.E           for hit in self.hits])
        self.time             = np.array([hit.time        for hit in self.hits])
        self.pathlength       = np.array([hit.pathlength  for hit in self.hits])
        self.quality          = np.array([hit.quality     for hit in self.hits])
        self.x                = np.array([hit.x           for hit in self.hits])
        self.y                = np.array([hit.y           for hit in self.hits])
        self.z                = np.array([hit.z           for hit in self.hits])
        self.px               = np.array([hit.px          for hit in self.hits])
        self.py               = np.array([hit.py          for hit in self.hits])
        self.pz               = np.array([hit.pz          for hit in self.hits])
        self.r                = np.array([hit.r           for hit in self.hits])
        self.theta            = np.array([hit.theta       for hit in self.hits])
        self.phi              = np.array([hit.phi         for hit in self.hits])

    def __iter__(self):
        for hit in self.hits:
            yield hit

    def __getitem__(self, index):
        return self.hits[index]

    def __len__(self):
        return self.N


class SimCalorimeterHitCollection:
    def __init__(self, rawhits):
        self.hits             = np.array(rawhits)
        self.N                = len(rawhits)
        self.cellID           = np.array([hit.cellID  for hit in self.hits])
        self.E                = np.array([hit.E       for hit in self.hits])
        self.x                = np.array([hit.x       for hit in self.hits])
        self.y                = np.array([hit.y       for hit in self.hits])
        self.z                = np.array([hit.z       for hit in self.hits])
        self.r                = np.array([hit.r       for hit in self.hits])
        self.theta            = np.array([hit.theta   for hit in self.hits])
        self.phi              = np.array([hit.phi     for hit in self.hits])

    def __iter__(self):
        for hit in self.hits:
            yield hit

    def __getitem__(self, index):
        return self.hits[index]

    def __len__(self):
        return self.N


# ---------------------------------------------------------------------------
# Registry — looked up by name from config
# ---------------------------------------------------------------------------
HIT_TYPES = {
    "SimTrackerHit": (SimTrackerHit, SimTrackerHitCollection),
    "SimCalorimeterHit": (SimCalorimeterHit, SimCalorimeterHitCollection),
}


