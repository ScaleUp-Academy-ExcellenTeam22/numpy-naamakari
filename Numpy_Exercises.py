from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt


# ex1
def change_sign(array: np, start: int, end: int) -> np:
    """
    The function changes the sign in a certain range in the vector it receives.
    :param array: The vector in him needs to change the sign.
    :param start: The beginning of the range.
    :param end: The end of the range.
    :return: The vector after the change.
    """
    array[(array >= start) & (array <= end)] *= -1
    return array


# ex1
def create_vector(start: int, end: int) -> np:
    """
    The function creates a vector in a certain range according to the range it receives
    and returns the vector it created.
    :param start: The beginning of the range.
    :param end: The end of the range.
    :return: The created vector.
    """
    return np.arange(start, end+1)


# ex2
def vector_with_values_evenly_distributed(start: int, end: int, space: int) -> np:
    """
    The function creates a vector in a certain range with equal spaces between them.
    :param start: The beginning of the range.
    :param end: The end of the range.
    :param space: The space between the values
    :return: The created vector.
    """
    return np.linspace(start, end, space)


# ex3
def find_row_and_columns(matrix: np) -> Tuple:
    """
    The function receives a matrix and returns its rows and columns.
    :param matrix: The matrix whose columns and rows want to know.
    :return: Tuple of the rows and columns of the received matrix
    """
    return matrix.shape[:2]


# ex4
def create_matrix_with_frame(matrix_size: int) -> np:
    """
    The function creates a matrix whose frame has 1 and all other places have 0.
    :param matrix_size: The size of the matrix you want to create.
    :return: The matrix created.
    """
    array = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    array[:1] = 1
    array[-1:] = 1
    array[:-1, :1] = 1
    array[:-1, -1:] = 1
    return array


# ex5
def add_vector_to_matrix(matrix: np, vector: np) -> np:
    """
    The function gets a vector and matrix and adds the vector to each row in the matrix.
    :param matrix: The matrix received.
    :param vector: The vector received.
    :return: The matrix after the change.
    """
    for row in range(matrix.shape[0]):
        matrix[row] += vector
    return matrix


# ex6
def show_sin_graph() -> None:
    """
    The function creates a sine graph and displays it.
    """
    x = np.arange(0, 5 * np.pi, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()


# ex7
def swapping_first_and_last_row_in_matrix(matrix: np) -> np:
    """
    The function switches between the first and last row in the received matrix.
    :param matrix: The received matrix.
    :return: The matrix after the change.
    """
    new_matrix = matrix[:-2:-1]
    new_matrix = np.vstack([new_matrix, matrix[1:-1:]])
    new_matrix = np.vstack([new_matrix, matrix[:1:]])
    return new_matrix


# ex8
def replace_number_in_array(array8: np, number_in_matrix: int, number_to_insert_instead: int, equal_sign: str) -> np:
    """
    The function replaces all the numbers in a given array that is equal, small and large to a given number,
    depending on the received sign.
    :param array8: The received array.
    :param number_in_matrix: The number to which we compare the values in the array.
    :param number_to_insert_instead: The number to which we will replace if the current value is
    small, large or equal to the received value.
    :param equal_sign: The value with which we make the comparison can be small, large or equal.
    :return: The array after the change.
    """
    if equal_sign == '=':
        return np.where(array8 == number_in_matrix, number_to_insert_instead, array8)
    if equal_sign == '<':
        return np.where(array8 < number_in_matrix, number_to_insert_instead, array8)
    if equal_sign == '>':
        return np.where(array8 > number_in_matrix, number_to_insert_instead, array8)


# ex9
def multiple_2_arrays(array1: np.array, array2: np.array) -> np.array:
    """
    The function takes 2 vectors and multiplies them, element by element.
    :param array1: The first vector.
    :param array2: The second vector.
    :return: The newly created vector.
    """
    return np.multiply(array1, array2)


# ex10
def sort_array_along_selected_axis(array10: np.array, axis: str) -> np.array:
    """
    The function receives a array and sorts it according to the received axis.
    :param array10: The received array.
    :param axis: The axis by which to sort the received vector.
    :return: Sorted array.
    """
    if axis == 'x':
        return np.sort(array10, axis=0)
    if axis == 'y':
        x = np.sort(array10, axis=0)
        return np.sort(x, axis=1)


# ex11
def create_3d_identity_array(matrix_size: int) -> np.array:
    """
    The function gets a number and creates the identity matrix the size of that number.
    :param matrix_size: The size of the matrix.
    :return: The newly created array.
    """
    return np.eye(matrix_size)


# ex12
def remove_single_dimensional_entries(array12: np.array) -> np.array:
    """
    The function removes one-dimensional values from a specified form.
    :param array12: The vector from which you want to remove the one-dimensional values.
    :return: The vector after removal.
    """
    return np.squeeze(array12).shape


# ex13
def convert_2_arrays_to_1(array1: np.array, array2: np.array) -> np.array:
    """
    The function converts (in depth sequence in terms (along the third axis)) two two-dimensional arrays
    into a two-dimensional array.
    :param array1: The first array.
    :param array2: The second array.
    :return: The array after the conversion.
    """
    return np.dstack((array1, array2))


# ex14
def combine_2_arrays_and_display_elements(array1: np.array, array2: np.array) -> None:
    """
    The function combines one and two-dimensional array together and displays their elements.
    :param array1: The first array.
    :param array2: The second array.
    """
    for x, y in np.nditer([array1, array2]):
        print(f"{x}:{y}")


if __name__ == '__main__':
    # ex1
    # print(change_sign(create_vector(0, 20), 9, 15))

    # ex2
    # print(vector_with_values_evenly_distributed(5, 50, 10))

    # ex3
    # array3 = np.arange(6).reshape(2, 3)
    # print(find_row_and_columns(array3))

    # ex4
    # print(create_matrix_with_frame(10))

    # ex5
    # original_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # vector_to_add = np.array([1, 1, 0])
    # print(add_vector_to_matrix(original_matrix, vector_to_add))

    # ex6
    # show_sin_graph()

    # ex7
    # original_matrix = np.random.randint(0, 100, (4, 4))
    # print(original_matrix)
    # print(swapping_first_and_last_row_in_matrix(original_matrix))

    # ex8
    # array8 = np.arange(16, dtype=np.int64).reshape(-1, 4)
    # print(array8)
    # print(replace_number_in_array(array8, 2, 12, '='))
    # print(replace_number_in_array(array8, 2, 12, '<'))
    # print(replace_number_in_array(array8, 2, 12, '>'))

    # ex9
    # array9first = np.array([[1, 2, 3], [3, 2, 1]])
    # array9second = np.array([[4, 5, 6], [6, 5, 4]])
    # print(multiple_2_arrays(array9first, array9second))

    # ex10
    # array10 = np.array([[4, 6], [2, 1]])
    # print(array10)
    # print(sort_array_along_selected_axis(array10, 'x'))
    # print(sort_array_along_selected_axis(array10, 'y'))

    # ex11
    # print(create_3d_identity_array(3))

    # ex12
    # array12 = np.ones((3, 1, 4))
    # print(remove_single_dimensional_entries(array12))

    # ex13
    # array13one = np.array([10, 20, 30])
    # array13two = np.array([40, 50, 60])
    # print(convert_2_arrays_to_1(array13one, array13two))

    # ex14
    array14one = np.array([0, 1, 2, 3])
    array14two = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    combine_2_arrays_and_display_elements(array14one, array14two)












