import openmesh as om
import numpy as np
from .geometry import quaternion_to_rotation_matrix


class Mesh:
    def __init__(self, filename=None, vertices=None, faces=None):
        if filename:
            self.topology = om.read_trimesh(filename)
        elif vertices is not None and faces is not None:
            self.topology = om.TriMesh()
            vh_list = []
            for v in vertices:
                vh = self.topology.add_vertex(v)
                vh_list.append(vh)
            for f in faces:
                face_vh = [vh_list[idx] for idx in f]
                self.topology.add_face(face_vh)
        else:
            self.topology = om.TriMesh()

    def save_obj(self, filename):
        om.write_mesh(filename, self.topology)
        print(f"save mesh to {filename}")

    def points(self):
        return self.topology.points()

    def face_vertex_indices(self):
        return self.topology.face_vertex_indices()

    def set_points(self, points):
        for vh, p in zip(self.topology.vertices(), points):
            self.topology.set_point(vh, p)

    def normalize(self):
        min_coord, max_coord = self.bbox()
        scale = 1.0 / (max_coord[1] - min_coord[1])
        center = (min_coord + max_coord) / 2.0
        half_height = (max_coord[1] - min_coord[1]) * scale / 2.0

        points = self.topology.points()
        points -= center
        points *= scale
        points[:, 1] += half_height
        self.set_points(points)

    def centered(self):
        bottom_center = self.bottom_center()
        points = self.topology.points()
        translated_points = points - bottom_center
        self.set_points(translated_points)

    def scale(self, scale_factor):
        points = self.topology.points()
        points *= scale_factor
        self.set_points(points)

    def apply_transform(self, quaternion=np.array([1.0, 0.0, 0.0, 0.0]), translation=np.array([0.0, 0.0, 0.0])):
        quaternion = quaternion / np.linalg.norm(quaternion)
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        points = self.topology.points()
        points = (rotation_matrix @ points.T).T + translation
        self.set_points(points)

    def apply_transformation_matrix(self, transform_matrix):
        points = np.hstack((self.topology.points(), np.ones((len(self.topology.points()), 1))))
        points = (transform_matrix @ points.T).T[:, :3]
        self.set_points(points)

    def lift_base(self, height=0.0):
        points = self.topology.points()
        min_y = np.min(points[:, 1])
        points[:, 1] += height - min_y
        self.set_points(points)

    def point(self, vh):
        return self.topology.point(vh)

    def he_angle(self, heh):
        if self.topology.is_boundary(heh):
            return 0.0
        vh_from = self.topology.from_vertex_handle(heh)
        vh_to = self.topology.to_vertex_handle(heh)
        next_heh = self.topology.next_halfedge_handle(heh)
        vh_next = self.topology.to_vertex_handle(next_heh)

        pa = self.topology.point(vh_from)
        pb = self.topology.point(vh_to)
        pc = self.topology.point(vh_next)

        u = pb - pa
        v = pc - pa

        u_norm = u / np.linalg.norm(u)
        v_norm = v / np.linalg.norm(v)

        dot_product = np.clip(np.dot(u_norm, v_norm), -1.0, 1.0)
        angle = np.arccos(dot_product)
        return angle

    def dual_area(self, vh):
        area = 0.0
        for fh in self.topology.vf(vh):
            area += self.face_area(fh)
        return area / 3.0

    def face_area(self, fh):
        vh_list = [vh for vh in self.topology.fv(fh)]
        p0 = self.topology.point(vh_list[0])
        p1 = self.topology.point(vh_list[1])
        p2 = self.topology.point(vh_list[2])

        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        return area

    def face_normal(self, fh):
        vh_list = [vh for vh in self.topology.fv(fh)]
        p0 = self.topology.point(vh_list[0])
        p1 = self.topology.point(vh_list[1])
        p2 = self.topology.point(vh_list[2])

        normal = np.cross(p1 - p0, p2 - p0)

        normal /= np.linalg.norm(normal)
        return normal

    def vertex_normal(self, vh):
        normal = np.zeros(3)
        for fh in self.topology.vf(vh):
            normal += self.face_normal(fh)

        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        else:
            normal = np.array([0.0, 1.0, 0.0])
        return normal

    def cotangent(self, heh):
        if self.topology.is_boundary(heh):
            return 0.0
        vh_from = self.topology.from_vertex_handle(heh)
        vh_to = self.topology.to_vertex_handle(heh)
        next_heh = self.topology.next_halfedge_handle(heh)
        vh_next = self.topology.to_vertex_handle(next_heh)

        pa = self.topology.point(vh_from)
        pb = self.topology.point(vh_to)
        pc = self.topology.point(vh_next)

        u = pa - pc
        v = pb - pc

        cross = np.cross(u, v)
        cross_norm = np.linalg.norm(cross)
        dot = np.dot(u, v)
        cotangent = dot / cross_norm if cross_norm != 0 else 0.0
        cotangent = max(cotangent, 1e-7)
        return cotangent

    def barycentric(self, fh):
        vh_list = [vh for vh in self.topology.fv(fh)]
        p0 = self.topology.point(vh_list[0])
        p1 = self.topology.point(vh_list[1])
        p2 = self.topology.point(vh_list[2])
        barycenter = (p0 + p1 + p2) / 3.0
        return barycenter

    def bbox(self):
        points = self.topology.points()
        min_coord = np.min(points, axis=0)
        max_coord = np.max(points, axis=0)
        return min_coord, max_coord

    def center(self):
        points = self.topology.points()
        center = np.mean(points, axis=0)
        return center

    def bottom_center(self, axis='y', epsilon=1e-2):
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis]
        points = self.topology.points()
        min_val = np.min(points[:, axis_idx])
        mask = np.abs(points[:, axis_idx] - min_val) < epsilon
        bottom_points = points[mask]
        if bottom_points.size == 0:
            return None
        avg_point = np.mean(bottom_points, axis=0)
        avg_point[axis_idx] = min_val
        return avg_point

    def compute_vertex_normals(self):
        self.topology.request_vertex_normals()
        self.topology.update_normals()
        return self.topology.vertex_normals()

    def compute_face_normals(self):
        self.topology.request_face_normals()
        self.topology.update_normals()
        return self.topology.face_normals()
