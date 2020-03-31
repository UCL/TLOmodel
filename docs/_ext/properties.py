# Based on "todo" Sphinx extension tutorial, https://www.sphinx-doc.org/en/master/development/tutorials/todo.html
from docutils import nodes
from docutils.parsers.rst import Directive

from sphinx.locale import _
from sphinx.util.docutils import SphinxDirective


#  properties inherits from Admonition because it should be handled like a note or warning,
class properties(nodes.Admonition, nodes.Element):
    pass


# propertieslist is just a “general” node.
class propertieslist(nodes.General, nodes.Element):
    pass


def visit_properties_node(self, node):
    self.visit_admonition(node)


def depart_properties_node(self, node):
    self.depart_admonition(node)


class propertieslistDirective(Directive):

    def run(self):
        return [propertieslist('')]


class propertiesDirective(SphinxDirective):

    # this enables content in the directive
    has_content = True

    def run(self):
        targetid = 'properties-%d' % self.env.new_serialno('properties')
        targetnode = nodes.target('', '', ids=[targetid])

        properties_node = properties('\n'.join(self.content))
        properties_node += nodes.title(_('properties'), _('properties'))
        self.state.nested_parse(self.content, self.content_offset, properties_node)

        if not hasattr(self.env, 'properties_all_properties'):
            self.env.properties_all_properties = []

        self.env.properties_all_properties.append({
            'docname': self.env.docname,
            'lineno': self.lineno,
            'properties': properties_node.deepcopy(),
            'target': targetnode,
        })

        return [targetnode, properties_node]


def purge_properties(app, env, docname):
    if not hasattr(env, 'properties_all_properties'):
        return

    env.properties_all_properties = [properties for properties in env.properties_all_properties
                          if properties['docname'] != docname]


def process_properties_nodes(app, doctree, fromdocname):
    if not app.config.properties_include_properties:
        for node in doctree.traverse(properties):
            node.parent.remove(node)

    # Replace all propertieslist nodes with a list of the collected properties.
    # Augment each properties with a backlink to the original location.
    env = app.builder.env

    # From the bugfix
    if not hasattr(env, 'properties_all_properties'):
        #env.todo_all_todos = []
        env.properties_all_properties = []

    for node in doctree.traverse(propertieslist):
        if not app.config.properties_include_properties:
            node.replace_self([])
            continue

        content = []

        for properties_info in env.properties_all_properties:
            para = nodes.paragraph()
            filename = env.doc2path(properties_info['docname'], base=None)
            description = (
                _('(The original entry is located in %s, line %d and can be found ') %
                (filename, properties_info['lineno']))
            para += nodes.Text(description, description)

            # Create a reference
            newnode = nodes.reference('', '')
            innernode = nodes.emphasis(_('here'), _('here'))
            newnode['refdocname'] = properties_info['docname']
            newnode['refuri'] = app.builder.get_relative_uri(
                fromdocname, properties_info['docname'])
            newnode['refuri'] += '#' + properties_info['target']['refid']
            newnode.append(innernode)
            para += newnode
            para += nodes.Text('.)', '.)')

            # Insert into the propertieslist
            content.append(properties_info['properties'])
            content.append(para)

        node.replace_self(content)


def setup(app):
    app.add_config_value('properties_include_properties', False, 'html')

    app.add_node(propertieslist)
    app.add_node(properties,
                 html=(visit_properties_node, depart_properties_node),
                 latex=(visit_properties_node, depart_properties_node),
                 text=(visit_properties_node, depart_properties_node))

    app.add_directive('properties', propertiesDirective)
    app.add_directive('propertieslist', propertieslistDirective)
    app.connect('doctree-resolved', process_properties_nodes)
    app.connect('env-purge-doc', purge_properties)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }

