
import numpy as np
import InSituLinearSolvers as isls
import math


def getData( A_path, B_path ):
    A = np.loadtxt( A_path )
    B = np.loadtxt( B_path )
    nr,nc = A.shape
    return( (A,B) )

def doColSpaceSoln(A,B):
    # A deep copy is used to evaluate solution, as A and B will be modified.
    Acopy = np.copy(A)
    Bcopy = np.copy(B)
    z = isls.LinearSolver()
    z.setData(A,B)
    z.doCGS_real_col(niter=2)
    X = z.getSolution_col()
    diff = np.dot(Acopy,X) - Bcopy
    print('X (col)=', X)
#    print( 'Element-wise residue of Ax-b =', diff)
    print( 'RMSE (col)=', math.sqrt(sum(x**2 for x in diff)))   


def doRowSpaceSoln(A,B):
    # A deep copy is used to evaluate solution, as A and B will be modified.
    Acopy = np.copy(A)
    Bcopy = np.copy(B)
    z = isls.LinearSolver()
    z.setData(A,B)
    z.doCGS_real_row(niter=2)
#    z.doMGS_real_row(niter=2)
#    z.doMGS_persson_row(niter=2)
    X = z.getSolution_row()
    diff = np.dot(Acopy,X) - Bcopy
    print('X (row)=', X)
#    print( 'Element-wise residue of Ax-b =', diff)
    print( 'RMSE (row)=', math.sqrt(sum(x**2 for x in diff)))    

def showData(A,B):
    print('A-------')
    print(A)
    print('B-------')
    print(B)
    

# =================================================
# case1 = unique solution
# case2 = unique solution, over-specified (consistent)
# case3 = non-unique solution, under-specified

print('\ncase1 --------------------------------')
A,B = getData( 'data/case1_A_4x4.dat', 'data/case1_B_4x1.dat' )
doColSpaceSoln(np.copy(A), np.copy(B))
doRowSpaceSoln(np.copy(A), np.copy(B))

print('\ncase2 --------------------------------')
A,B = getData( 'data/case2_A_4x3.dat', 'data/case2_B_4x1.dat' )
doColSpaceSoln(np.copy(A), np.copy(B))
doRowSpaceSoln(np.copy(A), np.copy(B))

print('\ncase3 --------------------------------')
A,B = getData( 'data/case3_A_3x4.dat', 'data/case3_B_3x1.dat' )
doColSpaceSoln(np.copy(A), np.copy(B))
doRowSpaceSoln(np.copy(A), np.copy(B))




