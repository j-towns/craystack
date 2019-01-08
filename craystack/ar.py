import craystack as cs


def AutoRegressive(data_to_append, data_to_pops, data_init):
    def append(message, data):
        append_ = data_to_append(data)
        return append_(message, data)

    def pop(message):
        data = data_init()
        for data_to_pop in data_to_pops:
            pop_ = data_to_pop(data)
            message, data = pop_(message)
        return message, data
