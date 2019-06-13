============
Contributing
============

We'll flesh out the guidelines here further as the project gets going!

Bug reports
===========

When `reporting a bug <https://github.com/UCL/TLOmodel/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Development
===========

To set up `TLOmodel` for local development:

1. Fork `TLOmodel <https://github.com/UCL/TLOmodel>`_
   (look for the "Fork" button).
   If you have write access to this main repository, you can skip this step and clone
   it directly in step 2.
2. Clone your fork locally::

    git clone git@github.com:your_name_here/TLOmodel.git

3. Create a branch for local development::

    git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

4. When you're done making changes, run all the checks, doc builder and spell checker with `tox <http://tox.readthedocs.io/en/latest/install.html>`_ one command::

    tox

5. Commit your changes and push your branch to GitHub::

    git add .
    git commit  # Write a description of your changes in the editor and save
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``) [1]_.
2. Update documentation when there's new API, functionality etc.
3. Add yourself to ``authors.rst``.

.. [1] If you don't have all the necessary python versions available locally you can rely on Travis - it will
       `run the tests <https://travis-ci.com/UCL/TLOmodel/pull_requests>`_ for each change you add in the pull request.

       It will be slower though ...

Tips
----

To run a subset of tests::

    tox -e py36 -- pytest -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``)::

    detox

