class FrustumProjection:
    def __init__(self, K, R, t):
        """
        Initializes the FrustumProjection class with camera intrinsic and extrinsic parameters.
        K: Intrinsic matrix (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        """
        self.K = np.array(K)  # (3, 3)
        self.R = np.array(R)  # (3, 3)
        self.t = np.array(t)  # (3, 1)
        self.inv_K = np.linalg.inv(self.K)  # Inverse of the intrinsic matrix (3, 3)

    def bbox_to_frustum(self, bbox, near=0.1, far=10.0):
        """
        Projects the 2D bounding box into a 3D frustum by starting rays from each bbox corner.
        bbox: Bounding box in format [x1, y1, x2, y2]
        near: Near plane distance
        far: Far plane distance
        Returns the vertices of the frustum in world coordinates.
        """
        x1, y1, x2, y2 = bbox  # Bbox corners

        # Create the 8 points for the frustum in normalized camera coordinates
        corners = np.array([
            [x1, y1, 1], [x2, y1, 1],
            [x2, y2, 1], [x1, y2, 1]
        ])  # (4, 3)

        # Back-project to camera coordinates
        rays = np.dot(self.inv_K, corners.T).T  # (4, 3)

        # Extend rays into the 3D world by using near and far planes
        frustum_vertices_near = self.extend_rays(rays, near)  # (4, 3)
        frustum_vertices_far = self.extend_rays(rays, far)  # (4, 3)

        # Combine near and far vertices
        frustum_vertices = np.vstack((frustum_vertices_near, frustum_vertices_far))  # (8, 3)

        return frustum_vertices

    def extend_rays(self, rays, depth):
        """
        Extends the rays from the camera center into the 3D world.
        rays: Rays in camera coordinates (4, 3)
        depth: Depth value (scalar)
        Returns the extended 3D points in world coordinates.
        """
        # Convert to homogeneous coordinates by multiplying with depth
        extended_rays = rays * depth  # (4, 3)

        # Transform from camera coordinates to world coordinates
        world_points = np.dot(self.R, extended_rays.T).T + self.t.T  # (4, 3)

        return world_points


class Mesh:
    def __init__(self):
        self.meshes = []

    def add_frustum(self, frustum_vertices):
        """
        Adds a frustum's vertices to the mesh.
        frustum_vertices: Vertices of the frustum (8, 3)
        """
        self.meshes.append(frustum_vertices)

    def intersect_meshes(self):
        """
        Computes the intersection of all meshes added so far.
        Returns the intersection points if more than one mesh exists.
        """
        if len(self.meshes) < 2:
            return None

        intersection = self.meshes[0]  # Start with the first mesh
        for mesh in self.meshes[1:]:
            combined_points = np.vstack((intersection, mesh))  # Combine points (N, 3)
            hull_combined = ConvexHull(combined_points)  # Compute convex hull
            intersection = hull_combined.points[hull_combined.vertices]  # (M, 3)

        return intersection

    def compute_frustum_faces(self, frustum_vertices):
        """
        Constructs the 12 triangles forming the faces of the frustum.
        frustum_vertices: Vertices of the frustum (8, 3)
        Returns a list of triangles representing the frustum faces.
        """
        faces = [
            [0, 1, 2], [0, 2, 3],  # Near plane
            [4, 5, 6], [4, 6, 7],  # Far plane
            [0, 1, 5], [0, 5, 4],  # Side 1
            [1, 2, 6], [1, 6, 5],  # Side 2
            [2, 3, 7], [2, 7, 6],  # Side 3
            [3, 0, 4], [3, 4, 7]   # Side 4
        ]
        return faces

    def visualize_mesh(self, vertices, title="Mesh Visualization"):
        """
        Visualizes the mesh vertices in a 3D plot.
        vertices: Vertices of the mesh (N, 3)
        title: Title of the plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
        plt.title(title)
        plt.show()