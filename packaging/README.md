# Publishing eventhorizon

Both packages are built and tested from the `v1.0.0` tag: `debian/` makes
`eventhorizon_1.0.0-1_all.deb`, and `packaging/arch/PKGBUILD` makes the Arch
package. Each runs the full test suite during the build.

## Arch (AUR) - live as soon as it is pushed

The AUR is where Arch users, Omarchy included, get community packages
(`yay -S eventhorizon`). There is no review queue.

1. Make an account at https://aur.archlinux.org/register.
2. In *My Account*, paste the public SSH key, `~/.ssh/id_ed25519.pub`.
3. Push the package, which creates it:

       git clone ssh://aur@aur.archlinux.org/eventhorizon.git ~/aur/eventhorizon
       cp packaging/arch/PKGBUILD packaging/arch/.SRCINFO ~/aur/eventhorizon/
       cd ~/aur/eventhorizon && git add PKGBUILD .SRCINFO
       git commit -m "eventhorizon 1.0.0" && git push

For a new version, bump `pkgver` (and reset `pkgrel=1`), update `sha256sums`
from the new tag's tarball, regenerate `.SRCINFO` with
`makepkg --printsrcinfo > .SRCINFO`, and push again.

## Debian - a .deb now, the archive later

**Now:** the `.deb` is attached to the GitHub release. Anyone on Debian or
Ubuntu can install it with

    sudo apt install ./eventhorizon_1.0.0-1_all.deb

**Into Debian itself** takes a sponsor, because only Debian Developers can
upload. Expect weeks to months; nothing here is automatic.

1. **Intent to package.** File a bug against the `wnpp` pseudo-package saying
   you are packaging eventhorizon: `reportbug wnpp`, choose ITP. The bug number
   comes back by email.
2. **Close it in the changelog.** In `debian/changelog`, change
   `* Initial release.` to `* Initial release. (Closes: #NNNNNN)`, and rebuild.
3. **Upload to mentors.** Make an account at https://mentors.debian.net, add
   the GPG key (DC128042), sign the source package with
   `debsign eventhorizon_1.0.0-1_source.changes`, and upload with
   `dput mentors eventhorizon_1.0.0-1_source.changes`.
4. **Ask for a sponsor.** File a bug against `sponsorship-requests` (RFS). The
   mentors site generates the text. A Debian Developer reviews it, may ask for
   changes, and uploads it.
5. **NEW queue.** The ftpmasters check licensing and packaging. Once accepted it
   is in unstable, moves to testing, and is in the next Debian release.

Likely review questions: the two-letter `/usr/bin/eh` (generic short names
attract objections; dropping it from the Debian package is an easy
concession), and that the package installs a Python module called `luminet`.
