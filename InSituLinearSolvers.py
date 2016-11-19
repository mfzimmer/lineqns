
import numpy as np

"""
Class is meant to be used once, to solve Ax=b using either a row or column based approach.
The test program demonstrates usage.

Author: Michael Zimmer
Date: 11/18/2016
"""

class LinearSolver:

    def setData(self,mA,mB,tol = 0.00001):
        self.eps_tol = tol
        self.A = mA
        self.B = mB
        self.nrowA, self.ncolA = (self.A).shape
        self.M = np.identity(self.ncolA)
#        print( 'A rows = ', self.nrowA)
#        print( 'A cols = ', self.ncolA)
        # Expecting B to be a 1-dim, column matrix.  Can generalize later.
        assert len((self.B).shape) == 1
        assert (self.B).shape[0] == self.nrowA

# ========================================================
    def normV_col(self,j):
#        "Expect i = 0,1,...,nrowA-1"
        assert j >= 0 & j < self.ncolA
#        print( 'normV, j = ', j)
        sum = np.sqrt( np.dot(self.A.T[j], self.A.T[j]) )
        if sum < self.eps_tol:
#            print( 'col #', j, ' is effectively zero.  sum=', sum)
#            print( 'setting it to exactly zero')
            self.A.T[j] = 0.0
        else:
            self.A.T[j] /= sum
            self.M.T[j] /= sum

    def doCGS_real_col(self, niter=2):
#        "MGS: niter = number of iterations of GS procedure"
        for n in range(niter):
#            print( 'MGS: niter= ', n)
            for j in range(self.ncolA):
#                print( 'j = ', j)
                rA = np.zeros(self.nrowA)
                rB = 0.0  #When B = matrix, change to rB=np.zeros(ncolB)
                for j2 in range(0,j):
#                    print( ' j2 = ', j2)
                    d = np.dot( self.A.T[j], self.A.T[j2] ) 
                    rA += d * self.A.T[j2]
                    rB += d * self.M.T[j2]
                self.A.T[j] -= rA
                self.M.T[j] -= rB
                self.normV_col(j)

    def getSolution_col(self):
        G = np.dot( self.M, self.A.T)
        return( np.dot(G,self.B) )

# ========================================================
    def normV_row(self,i):
#        "Expect i = 0,1,...,nrowA-1"
        assert i >= 0 & i < self.nrowA
#        print( 'normV, i = ', i)
        sum = np.sqrt( np.dot(self.A[i], self.A[i]) )
        if sum < self.eps_tol:
#            print( 'row #', i, ' is effectively zero.  sum=', sum)
#            print( 'setting it to exactly zero')
            self.A[i] = 0.0
        else:
            self.A[i] /= sum
            self.B[i] /= sum

    def doCGS_real_row(self, niter=2):
#        "MGS: niter = number of iterations of GS procedure"
        for n in range(niter):
#            print( 'MGS: niter= ', n)
            for i in range(self.nrowA):
#                print( 'i = ', i)
                rA = np.zeros(self.ncolA)
                rB = 0.0  #When B = matrix, change to rB=np.zeros(ncolB)
                for i2 in range(0,i):
#                    print( ' i2 = ', i2)
                    d = np.dot( self.A[i], self.A[i2] ) 
                    rA += d * self.A[i2]
                    rB += d * self.B[i2]
                self.A[i] -= rA
                self.B[i] -= rB
                self.normV_row(i)

    def doMGS_real_row(self, niter=2):
#        "CGS: niter = number of iterations of GS procedure"
        for n in range(niter):
#            print( 'CGS: niter= ', n)
            for i in range(self.nrowA):
#                print( 'i = ', i)
                for i2 in range(0,i):
#                    print( ' i2 = ', i2)
                    d = np.dot( self.A[i], self.A[i2] )
                    self.A[i] -= d*self.A[i2]
                    self.B[i] -= d*self.B[i2]
                self.normV_row(i)

    def doMGS_persson_row(self, niter=2):
#        "CGS: niter = number of iterations of GS procedure"
        for n in range(niter):
#            print( 'CGS: niter= ', n)
            for i in range(self.nrowA):
#                print( 'i = ', i)
                self.normV_row(i)
                for i2 in range(i+1,self.nrowA):
#                    print( ' i2 = ', i2)
                    d = np.dot( self.A[i], self.A[i2] )
                    self.A[i2] -= d*self.A[i]
                    self.B[i2] -= d*self.B[i]

    def getSolution_row(self):
        return np.dot( (self.A).T, self.B)


