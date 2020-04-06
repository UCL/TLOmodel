# Based on "todo" Sphinx extension tutorial,
# https://www.sphinx-doc.org/en/master/development/tutorials/todo.html
from docutils import nodes
from docutils.parsers.rst import Directive

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


#  parameters inherits from Admonition because
#  it should be handled like a note or warning,
class parameters(nodes.Admonition, nodes.Element):
    pass


# parameterslist is just a “general” node.
class parameterslist(nodes.General, nodes.Element):
    pass


def visit_parameters_node(self, node):
    self.visit_admonition(node)


def depart_parameters_node(self, node):
    self.depart_admonition(node)


class parameterslistDirective(Directive):

    def run(self):
        return [parameterslist('')]


class parametersDirective(SphinxDirective):

    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = 'parameters-%d' % self.env.new_serialno('parameters')
        targetnode = nodes.target('', '', ids=[targetid])

        parameters_node = parameters('\n'.join(self.content))
        parameters_node += nodes.title(_('parameters'), _('parameters'))
        self.state.nested_parse(self.content, self.content_offset,
                                parameters_node)

        if not hasattr(self.env, 'parameters_all_parameters'):
            self.env.parameters_all_parameters = []

        self.env.parameters_all_parameters.append({
            'docname': self.env.docname,
            'lineno': self.lineno,
            'parameters': parameters_node.deepcopy(),
            'target': targetnode,
        })

        return [targetnode, parameters_node]


def purge_parameters(app, env, docname):
    if not hasattr(env, 'parameters_all_parameters'):
        return

    env.parameters_all_parameters = [parameters
                                     for parameters
                                     in env.parameters_all_parameters
                                     if parameters['docname'] != docname]


def process_parameters_nodes(app, doctree, fromdocname):
    if not app.config.parameters_include_parameters:
        for node in doctree.traverse(parameters):
            node.parent.remove(node)

    # Replace all parameterslist nodes with a list of the collected parameters.
    # Augment each parameters with a backlink to the original location.
    env = app.builder.env

    # From the bugfix
    if not hasattr(env, 'parameters_all_parameters'):
        env.parameters_all_parameters = []

    for node in doctree.traverse(parameterslist):
        if not app.config.parameters_include_parameters:
            node.replace_self([])
            continue

        content = []

        for parameters_info in env.parameters_all_parameters:
            para = nodes.paragraph()
            filename = env.doc2path(parameters_info['docname'], base=None)
            description = (
                _('(The original entry is located in %s, line %d '
                  'and can be found ') %
                (filename, parameters_info['lineno']))
            para += nodes.Text(description, description)

            # Create a reference
            newnode = nodes.reference('', '')
            innernode = nodes.emphasis(_('here'), _('here'))
            newnode['refdocname'] = parameters_info['docname']
            newnode['refuri'] = app.builder.get_relative_uri(
                fromdocname, parameters_info['docname'])
            newnode['refuri'] += '#' + parameters_info['target']['refid']
            newnode.append(innernode)
            para += newnode
            para += nodes.Text('.)', '.)')

            # Insert into the parameterslist
            content.append(parameters_info['parameters'])
            content.append(para)

        node.replace_self(content)


def setup(app):
    app.add_config_value('parameters_include_parameters', False, 'html')

    app.add_node(parameterslist)
    app.add_node(parameters,
                 html=(visit_parameters_node, depart_parameters_node),
                 latex=(visit_parameters_node, depart_parameters_node),
                 text=(visit_parameters_node, depart_parameters_node))

    app.add_directive('parameters', parametersDirective)
    app.add_directive('parameterslist', parameterslistDirective)
    app.connect('doctree-resolved', process_parameters_nodes)
    app.connect('env-purge-doc', purge_parameters)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
