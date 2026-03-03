"""Handle cubic splines - a temporary fudge until arrays are sorted out."""

import math

from  .CubicSpline import CubicSpline
from ..Arrays      import Array

#===================================================================================================================================
# . Parameters.
#===================================================================================================================================
# . Gaussian quadrature parameters - 8 point rule (points, weights).
_GQData = ( (  0.183434642495650, 0.362683783378362 ) ,
            (  0.525532409916329, 0.313706645877887 ) ,
            (  0.796666477413627, 0.222381034453374 ) ,
            (  0.960289856497536, 0.101228536290376 ) ,
            ( -0.960289856497536, 0.101228536290376 ) ,
            ( -0.796666477413627, 0.222381034453374 ) ,
            ( -0.525532409916329, 0.313706645877887 ) ,
            ( -0.183434642495650, 0.362683783378362 ) )

# . Curvature tolerance.
_TOLERANCE = 1.0e-7

#===================================================================================================================================
# . Class.
#===================================================================================================================================
class MultiCubicSpline:
    """Multicubic splines."""

    def __init__ ( self, *arguments, **keywordArguments ):
        """Constructor."""
        # . Initialization.
        self.arcLengths = None
        self.splines    = []
        # . X and Y values.
        if len ( arguments ) > 1:
            x = arguments[0]
            y = arguments[1]
        else:
            x = None
            y = arguments[0]
        # . Values.
        npoints  = len ( y    )
        nsplines = len ( y[0] )
        # . Get local keyword arguments.
        localKeywordArguments = dict ( keywordArguments )
        localKeywordArguments["x"] = x
        # . Calculate the individual splines.
        # . Y is a list of vectors or similar.
        for ispline in range ( nsplines ):
            yValues = []
            for i in range ( npoints ): yValues.append ( y[i][ispline] )
            self.splines.append ( CubicSpline.FromFunctionValues ( yValues, **localKeywordArguments ) )

    def AbscissaValue ( self, i ):
        """Return an abscissa value."""
        if len ( self.splines ) > 0: return self.splines[0].AbscissaValue ( i )
        else: raise IndexError ( "Point index out of range." )

    def ArcLength ( self, x0, x1, g = None ):
        """Calculate the arc length between two abscissa values."""
        # . Find the abscissa starting and stopping values (do by segment).
        segmentData = []
        iLower      = self.Locate ( x0 )
        iUpper      = self.Locate ( x1 )
        for iSegment in range ( iLower, iUpper + 1 ):
            if iSegment == iLower: xStart = x0
            else:                  xStart = self.AbscissaValue ( iSegment     )
            if iSegment == iUpper: xStop  = x1
            else:                  xStop  = self.AbscissaValue ( iSegment + 1 )
            segmentData.append ( ( xStart, xStop ) )
        # . Allocate space.
        if g is None:
            g = Array.WithExtent ( self.NumberOfSplines ( ) )
            g.Set ( 0.0 )
        # . Do the integration of gNorm2 for each of the segments (using Gaussian quadrature).
        arcLength = 0.0
        for ( xStart, xStop ) in segmentData:
            c = 0.5 * ( xStart + xStop  )
            f = 0.5 * ( xStop  - xStart )
            for ( p, w ) in _GQData:
                arcLength += ( f * w * self.GradientNorm2 ( c + f * p, g = g ) )
        return arcLength

    def Curvature ( self, x, g = None, h = None ):
        """Calculate the curvature."""
        # . Definition?
        if g is None:
            g = Array.WithExtent ( self.NumberOfSplines ( ) )
            g.Set ( 0.0 )
        if h is None:
            h = Array.WithExtent ( self.NumberOfSplines ( ) )
            h.Set ( 0.0 )
        self.Evaluate ( x, g = g, h = h )
        gNorm2 = g.Norm2 ( )
        if gNorm2 > _TOLERANCE:
            g.Scale ( 1.0 / gNorm2 )
            ht = h.Dot ( g )
            h.Add ( g, scale = -ht )
            curvature = h.Norm2 ( ) / ( gNorm2**2 )
        else:
            curvature = 0.0
        return curvature

    def Distance ( self, x0, x1 ):
        """Return the distance."""
        # . Where does this come from - like an RMS?
        return self.ArcLength ( x0, x1 ) / math.sqrt ( self.NumberOfSplines ( ) )

    def DetermineArcLengths ( self ):
        """Calculate the arc lengths for each segment."""
        if self.arcLengths is None:
            self.arcLengths = Array.WithExtent ( self.NumberOfPoints ( ) - 1 )
            self.arcLengths.Set ( 0.0 )
            for i in range ( self.NumberOfPoints ( ) - 1 ):
                self.arcLengths[i] = self.ArcLength ( self.AbscissaValue ( i ), self.AbscissaValue ( i + 1 ) )

    def Evaluate ( self, x, f = None, g = None, h = None ):
        """Evaluate the spline."""
        for ( i, spline ) in enumerate ( self.splines ):
            ( fx, gx, hx ) = spline.Evaluate ( x )
            if f is not None: f[i] = fx
            if g is not None: g[i] = gx
            if h is not None: h[i] = hx

    def FindAbscissaFromArcLength ( self, arcLength):
        """Find the value of the spline abscissa given an arc length."""
        # . Make sure the arc lengths exist.
        self.DetermineArcLengths ( )
        # . Check for bad values.
        if ( arcLength< 0.0 ) or ( arcLength > sum ( self.arcLengths ) ): raise ValueError ( "Invalid arc length value." )
        # . Determine the segment within which the arc length falls.
        iSegment = -1
        lower    = 0.0
        upper    = 0.0
        for i in range ( self.NumberOfPoints ( ) - 1 ):
            lower  = upper
            upper += self.arcLengths[i]
            if   arcLength == lower: return self.AbscissaValue ( i     )
            elif arcLength == upper: return self.AbscissaValue ( i + 1 )
            elif ( arcLength > lower ) and ( arcLength < upper ):
                iSegment = i
                break
        # . Search for a root with the secant method.
        # . Initialization (the function is arcLength - required arcLength).
        arcLength -= lower
        upper     -= lower
        xLower     = self.AbscissaValue ( iSegment     )
        xUpper     = self.AbscissaValue ( iSegment + 1 )
        a = xLower ; fa = - arcLength
        b = xUpper ; fb = upper - arcLength
        root = b
        # . Find the root.
        while math.fabs ( fb / upper ) > 1.0e-4:
            root   = b -  fb * ( b - a ) / ( fb - fa )
            length = self.ArcLength ( xLower, root )
            a = b    ; fa = fb
            b = root ; fb = length - arcLength
        return root

    def GradientNorm2 ( self, x, g = None ):
        """Calculate the gradient 2-norm."""
        if g is None:
            g = Array.WithExtent ( self.NumberOfSplines ( ) )
            g.Set ( 0.0 )
        self.Evaluate ( x, g = g )
        return g.Norm2 ( )

    def Locate ( self, x ):
        """Locate the segment within which the abscissa lies."""
        il = 0
        iu = self.NumberOfPoints ( ) - 1
        while ( iu - il ) > 1:
            im = ( iu + il ) // 2
            if x > self.AbscissaValue ( im ): il = im
            else:                             iu = im
        return il

    def Normal ( self, x, normal = None ):
        """Calculate the normal."""
        # . Second Frenet vector.
        g = Array.WithExtent ( self.NumberOfSplines ( ) )
        g.Set ( 0.0 )
        if normal is None:
            h = Array.WithExtent ( self.NumberOfSplines ( ) )
            h.Set ( 0.0 )
        else:
            h = normal
        self.Evaluate ( x, g = g, h = h )
        g.Normalize ( )
        ht = h.Dot ( g )
        h.Add ( g, scale = -ht )
        h.Normalize ( )
        return h

    def NumberOfPoints ( self ):
        """Return the number of abscissa points."""
        if len ( self.splines ) <= 0: return 0
        else:                         return self.splines[0].NumberOfPoints ( )

    def NumberOfSplines ( self ):
        """Return the number of splines."""
        return len ( self.splines )

    def Tangent ( self, x, tangent = None ):
        """Calculate the tangent."""
        # . First Frenet vector.
        if tangent is None:
            tangent = Array.WithExtent ( self.NumberOfSplines ( ) )
            tangent.Set ( 0.0 )
        self.Evaluate ( x, g = tangent )
        return tangent.Normalize ( )

    # . Properties.
    def __GetLower ( self ): return self.AbscissaValue ( 0 )
    def __GetUpper ( self ): return self.AbscissaValue ( self.NumberOfPoints ( ) - 1 )
    lower = property ( __GetLower, None, None, "Lower abscissa." )
    upper = property ( __GetUpper, None, None, "Upper abscissa." )

#===================================================================================================================================
# . Testing.
#===================================================================================================================================
if __name__ == "__main__" :
    pass
