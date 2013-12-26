
def plot(x, y ,z):
    #Xarr, Yarr = np.meshgrid(x,y)
    #
    #ax.plot_surface(Xarr, Yarr, z)
    #
    #plt.show()

    #import pylab as pl
    #pl.scatter([x[0] for x in X][:], y[:], c='k', label='data')
    #pl.hold('on')
    #predictions = list(clf.predict(X[:]))
    #predictions.insert(0,np.nan)
    #predictions.append(np.nan)
    #pl.plot([np.nan] + [x[0] for x in X][:] + [np.nan], predictions, c='g', label='Model')
    #pl.xlabel('data')
    #pl.ylabel('target')
    #pl.title(str(clf))
    #pl.xlim(0,24)
    #pl.ylim(0,100)
    #pl.legend()
    #pl.show()
    import numpy as np
    from mayavi import mlab


    x = np.array([tup[1] for tup in X])
    y = np.array(y)
    z = np.array([tup[0] for tup in X])

    # Define the points in 3D space
    # including color code based on Z coordinate.
    pts = mlab.points3d(x, y, z, y)

    # Triangulate based on X, Y with Delaunay 2D algorithm.
    # Save resulting triangulation.
    #mesh = mlab.pipeline.delaunay2d(pts)

    # Remove the point representation from the plot
    #pts.remove()

    #mesh = np.meshgrid(x, y, z)
    # Draw a surface based on the triangulation
    surf = mlab.pipeline.surface(pts)

    # Simple plot.
    mlab.xlabel("day of month")
    mlab.ylabel("# logins")
    mlab.zlabel("hour")
    mlab.savefig(filename='test.png')
    mlab.show()


    import ipdb; ipdb.set_trace()