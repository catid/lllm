/*

(c - MIT) T.W.J. de Geus (Tom) | tom@geus.me | www.geus.me | github.com/tdegeus/cppmat

*/

#ifndef CPPPATH_H
#define CPPPATH_H

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _MSC_VER
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

/**
\cond
*/
#define Q(x) #x
#define QUOTE(x) Q(x)
/**
\endcond
*/

/**
Library version.

Either:

-   Configure using CMake at install time. Internally uses:

        python -c "from setuptools_scm import get_version; print(get_version())"

-   Define externally using:

        -DCPPPATH_VERSION="..."

    From the root of this project. This is what ``setup.py`` does.

Note that both ``CMakeLists.txt`` and ``setup.py`` will construct the version using
``setuptools_scm``. Tip: use the environment variable ``SETUPTOOLS_SCM_PRETEND_VERSION`` to
overwrite the automatic version.
*/
#ifndef CPPPATH_VERSION
#define CPPPATH_VERSION "@PROJECT_VERSION@"
#endif

#ifndef CPPPATH_SEP
#ifdef _MSC_VER
#define CPPPATH_SEP "\\"
#else
#define CPPPATH_SEP "/"
#endif
#endif

namespace cpppath {

namespace detail {

/**
Remove " from string.

\param arg Input string.
\return String without "
*/
inline std::string unquote(const std::string& arg)
{
    std::string ret = arg;
    ret.erase(std::remove(ret.begin(), ret.end(), '\"'), ret.end());
    return ret;
}

} // namespace detail

/**
Return version string, for example `"0.1.0"`

\return std::string
*/
inline std::string version()
{
    return detail::unquote(std::string(QUOTE(CPPPATH_VERSION)));
}

// === OVERVIEW ===

// Get OS's separator.
// Unix: "/", Windows: "\\"

inline std::string sep();

// Get dirname part of a path.
// Depending on the path, an empty string may be returned.
// Example: "/path/to/foo/bar.txt" returns "/path/to/foo"

inline std::string dirname(const std::string& path, const std::string& sep = CPPPATH_SEP);

// Get filename part of a path.
// Depending on the path, an empty string may be returned.
// Example: "/path/to/foo/bar.txt" returns "bar.txt"

inline std::string filename(const std::string& path, const std::string& sep = CPPPATH_SEP);

// Get filename part of a path, without extension.
// Depending on the path, an empty string may be returned.
// Example: "/path/to/foo/bar.txt" returns "bar"

inline std::string filebase(
    const std::string& path,
    const std::string& sep = CPPPATH_SEP,
    const std::string& extsep = ".");

// Split the pathname path into a pair (root, ext) such that root + ext == path,
// and ext is empty or begins with a period and contains at most one period.
// Leading periods on the basename are ignored; splitext(".cshrc") returns {".cshrc", ""}.

inline std::vector<std::string> splitext(const std::string& path, const std::string& extsep = ".");

// Get the extension of a path.
// Depending on the path, an empty string may be returned.
// Example: "/path/to/foo/bar.txt" returns "txt"

inline std::string ext(const std::string& path, const std::string& extsep = ".");

// Join sub-paths together using the separator.
// Provides option to prepend the output string with the separator.

inline std::string
join(const std::vector<std::string>& paths, const std::string& sep = CPPPATH_SEP);

inline std::string
join(const std::vector<std::string>& paths, bool preprend, const std::string& sep = CPPPATH_SEP);

inline std::string join(const std::vector<std::string>& paths, const char* sep);

inline std::string join(const std::vector<std::string>& paths, bool preprend, const char* sep);

// Split sub-paths using the separator.
// Option: slice the output[begin: end]

inline std::vector<std::string>
split(const std::string& path, const std::string& sep = CPPPATH_SEP);

inline std::vector<std::string>
split(const std::string& path, int begin, int end = 0, const std::string& sep = CPPPATH_SEP);

// Select path of a path.
// Example: select("/path/to/foo/bar.txt", 2) returns "foo/bar.txt"
// Example: select("/path/to/foo/bar.txt", 2, 3) returns "foo"

inline std::string
select(const std::string& path, int begin, int end = 0, const std::string& sep = CPPPATH_SEP);

// Normalize a path by collapsing redundant separators and up-level references.

inline std::string normpath(const std::string& path, const std::string& sep = CPPPATH_SEP);

// Select the common prefix in a list of strings.

inline std::string commonprefix(const std::vector<std::string>& paths);

// Select the common dirname in a list of paths.

inline std::string commondirname(const std::vector<std::string>& paths);

// Return the current working directory.

inline std::string curdir();

// Check if a path exists.

inline bool exists(const std::string& path);

// === IMPLEMENATION ===

inline std::string sep()
{
    return CPPPATH_SEP;
}

inline std::string dirname(const std::string& path, const std::string& sep)
{
    size_t idx = path.find_last_of(sep);
    if (idx == std::string::npos) {
        return "";
    }
    return path.substr(0, idx);
}

inline std::string filename(const std::string& path, const std::string& sep)
{
    size_t idx = path.find_last_of(sep);
    if (idx == std::string::npos) {
        return path;
    }
    return path.substr(idx + 1, path.length());
}

inline std::string
filebase(const std::string& path, const std::string& sep, const std::string& extsep)
{
    std::string out = filename(path, sep);
    return splitext(out, extsep)[0];
}

inline std::vector<std::string> splitext(const std::string& path, const std::string& extsep)
{
    std::string e = ext(path, extsep);
    if (e.size() == 0) {
        return {path, ""};
    }
    return {path.substr(0, path.size() - e.size() - extsep.size()), e};
}

namespace detail {
inline bool all_extsep(const std::string& path, const std::string& extsep)
{
    if (extsep.size() != 1) {
        return false;
    }
    for (auto& i : path) {
        if (i != extsep[0]) {
            return false;
        }
    }
    return true;
}
} // namespace detail

inline std::string ext(const std::string& path, const std::string& extsep)
{
    size_t idx = path.find_last_of(extsep);
    if (idx == std::string::npos) {
        return "";
    }
    if (detail::all_extsep(path.substr(0, idx), extsep)) {
        return "";
    }
    return path.substr(idx + 1);
}

inline std::string join(const std::vector<std::string>& paths, const std::string& sep)
{
    if (paths.size() == 1) {
        return paths[0];
    }

    std::string out = "";

    for (auto path : paths) {

        if (out.size() == 0) {
            out += path;
            continue;
        }

        if (path[0] == sep[0]) {
            out += path;
        }
        else if (out[out.size() - 1] == sep[0]) {
            out += path;
        }
        else {
            out += sep + path;
        }
    }

    return out;
}

inline std::string
join(const std::vector<std::string>& paths, bool preprend, const std::string& sep)
{
    if (preprend) {
        return sep + join(paths, sep);
    }
    return join(paths, sep);
}

inline std::string join(const std::vector<std::string>& paths, const char* sep)
{
    return join(paths, std::string(sep));
}

inline std::string join(const std::vector<std::string>& paths, bool preprend, const char* sep)
{
    return join(paths, preprend, std::string(sep));
}

inline std::vector<std::string> split(const std::string& path, const std::string& sep)
{
    std::vector<std::string> out;

    size_t prev = 0;
    size_t pos = 0;

    do {
        // find next match (starting from "prev")
        pos = path.find(sep, prev);

        // no match found -> use length of string as 'match'
        if (pos == std::string::npos) {
            pos = path.length();
        }

        // get sub-string
        std::string token = path.substr(prev, pos - prev);

        // store sub-string in list
        if (!token.empty()) {
            out.push_back(token);
        }

        // move further
        prev = pos + sep.length();
    } while (pos < path.length() && prev < path.length());

    return out;
}

inline std::vector<std::string>
split(const std::string& path, int begin, int end, const std::string& sep)
{
    std::vector<std::string> paths = split(path, sep);

    int N = (int)paths.size();

    // automatically set the length

    if (end == 0) {
        end = N;
    }

    // convert negative indices that count from the end

    if (begin < 0) {
        begin = (N + (begin % N)) % N;
    }

    if (end < 0) {
        end = (N + (end % N)) % N;
    }

    // select path components

    std::vector<std::string> out;

    for (int i = begin; i < end; ++i) {
        out.push_back(paths[i]);
    }

    return out;
}

inline std::string select(const std::string& path, int begin, int end, const std::string& sep)
{
    std::string prefix = "";
    if (path[0] == sep[0]) {
        prefix = sep;
    }
    return prefix + join(split(path, begin, end), sep);
}

inline std::string normpath(const std::string& path, const std::string& sep)
{
    bool root = path[0] == sep[0];

    // list of path components (this removes already all "//")
    std::vector<std::string> paths = split(path, sep);

    // filter "."
    {
        std::vector<std::string> tmp;

        for (auto& i : paths) {
            if (i != ".") {
                tmp.push_back(i);
            }
        }

        paths = tmp;
    }

    // filter "foo/../"
    {
        while (true) {

            bool found = false;

            for (size_t i = 1; i < paths.size(); ++i) {

                if (paths[i] == "..") {

                    std::vector<std::string> tmp;

                    for (size_t j = 0; j < paths.size(); ++j) {
                        if (j != i && j != i - 1) {
                            tmp.push_back(paths[j]);
                        }
                    }

                    paths = tmp;
                    found = true;
                    break;
                }
            }

            if (!found) {
                break;
            }
        }
    }

    if (root) {
        return sep + join(paths, sep);
    }

    return join(paths, sep);
}

namespace detail {
inline bool all_equal(const std::vector<std::string>& paths, size_t i)
{
    for (size_t j = 1; j < paths.size(); ++j) {
        if (paths[0][i] != paths[j][i]) {
            return false;
        }
    }
    return true;
}
} // namespace detail

inline std::string commonprefix(const std::vector<std::string>& paths)
{
    if (paths.size() == 0) {
        return "";
    }

    size_t i;
    size_t n = paths[0].size();

    for (auto& path : paths) {
        if (path.size() < n) {
            n = path.size();
        }
    }

    for (i = 0; i < n; ++i) {
        if (detail::all_equal(paths, i)) {
            ++i;
        }
        else {
            break;
        }
    }

    return paths[0].substr(0, i);
}

inline std::string commondirname(const std::vector<std::string>& paths)
{
    return dirname(commonprefix(paths));
}

inline std::string curdir()
{
    char buff[FILENAME_MAX];
    char* out = GetCurrentDir(buff, FILENAME_MAX);
    if (out == NULL) {
        throw std::runtime_error("getcwd failed");
    }
    std::string cwd(buff);
    return cwd;
}

inline bool exists(const std::string& path)
{
    std::ifstream file(path);
    return static_cast<bool>(file);
}

} // namespace cpppath

#endif