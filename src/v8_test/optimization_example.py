import numpy
from scipy import optimize

def test_on_all_images(blue_color_as_lab, brown_color_as_lab):
    pass
    # ...
    # ...
    return average_kappa_score



blue_color_as_lab = (10, 10, 10)
brown_color_as_lab = (20, 20, 20)

test_on_all_images(blue_color_as_lab, brown_color_as_lab)


dictionary_arguments = {
    'blue_color_as_lab': (10, 10, 10),
    'brown_color_as_lab': (20, 20, 20),
}
test_on_all_images(**dictionary_arguments)


for param_name, param_initialvalue in dictionary_arguments.items():
    # optimize
    # get NEW best value
    # assign the new value
    pass








history = []

def wrapper_test_on_all_images(blue_color_as_lab):
    result = test_on_all_images(blue_color_as_lab)
    history.append({
        'parameter': blue_color_as_lab,
        'value': result,
    })
    print(f'The function with blue {blue_color_as_lab} had an kappa score of {result}.')
    return result


initial_blue_color = numpy.ndarray(50, 50, 50)

opt_result = optimize.minimize(wrapper_test_on_all_images, initial_blue_color)

print(f'The function with blue {opt_result.fun} had an kappa score of {opt_result.x}.')
