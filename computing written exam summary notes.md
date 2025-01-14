
basics
- The combination of hardware and software forms a usable computing system. Hardware is usually instructed by software to execute any commands or instructions. Computer hardware includes the physical, tangible parts or components of a computer, such as the main board, central processing unit (CPU), display, keyboard, and mouse. Computer software includes system software and application software. System software is responsible for managing the various independent hardware in the computer system so that they can work in coordination. Common operating systems in system software include Linux, macOS, Unix, Windows, etc. Application software is software developed for a specific purpose. Common application software includes word processing, programming, web browsers, input methods, and media players, etc.

ascii
- A bit is the smallest unit of data stored in a computer. It is either 0 or 1. A bit represents a certain state of a device. For example, 1 represents a switch that is closed and 0 represents a switch that is open. To represent different types of data, a bit pattern is used (sequence of 0s and 1s)
- 8 bit = 1 Byte
    - KB= 2^10 Byte
    - MB = 2^10 KB = 2^10 Byte
    - GB = 2^10 MB
    - TB = 2^10 GB
- 10: new line
- 13: 回⻋ return
- 48~57: numbers
- 65~90 : uppercase alphabet
- 97~122: lowercase alphabet
    - +(32)_10
    - +(00100000)_2
    - + (20)_16
- 127: delete

virtual memory
- Virtual memory is a memory management technique used in computers to extend the apparent amount of physical memory available by using disk space as an extension of RAM.
- Key Concepts of Virtual Memory:
    1. Address Space Virtualization:
        - Programs use a virtual address space that is larger than the physical RAM.
        - Each program thinks it has access to its entire contiguous block of memory.
    2. Paging:
        - Memory is divided into fixed-sized blocks called pages (typically 4 KB in size).
        - The corresponding storage on the disk is divided into page frames.
        - Only the required pages of a program are loaded into physical memory, not the entire program.
    3. Page Tables:
        - Each process has a page table to map virtual addresses to physical addresses.
        - A page table entry (PTE) contains metadata, including the frame location and access permissions.
    4. Swapping:
        - When a program requests data not in physical memory (a page fault), the operating system (OS) retrieves it from the disk.
        - If RAM is full, the OS swaps out the least-used page (or another policy-based page) to the disk to free up space.
    5. Memory Management Unit (MMU):
        - A hardware component in the CPU called the MMU translates virtual addresses to physical addresses using the page table.
    6. Disk Usage:
        - A portion of the hard drive or SSD is allocated as swap space to store pages not currently in RAM.
- Advantages of Virtual Memory:
    - Program Isolation: Each program operates within its own virtual memory space, enhancing security and stability.
    - Efficient Memory Utilization: Allows multiple programs to run simultaneously, even if their combined memory usage exceeds physical RAM.
    - Simplified Programming: Developers can write programs without worrying about the physical memory constraints.
- Challenges:
    - Performance Overhead: Accessing data from the disk is much slower than accessing RAM, leading to potential performance degradation (called thrashing) if the system relies too heavily on swapping.
    - Complexity: Managing virtual memory requires additional hardware (MMU) and software (OS algorithms).
- Key Processes in Virtual Memory:
    1. Page Fault Handling:
        - OS checks if the requested virtual page is in physical memory.
        - If not, it retrieves the page from disk and updates the page table.
    2. Replacement Algorithms (used when swapping pages):
        - Least Recently Used (LRU): Swaps out the page that hasn't been used for the longest time.
        - First-In, First-Out (FIFO): Swaps out the oldest page in memory.
        - Clock Algorithm: Uses a circular buffer to track page usage.
- Modern Uses:
    - Virtual memory is widely used in modern operating systems, such as Windows, Linux, and macOS, to efficiently manage resources and enable multitasking on computers with limited physical RAM.

turing machine
- A tape: infinitely long in both directions, with squares (fields) on it, each of which can contain a character from a finite alphabet. In a real machine, the tape must be large enough to contain all the data for the algorithm.
- A controller: contains a read-write head (head) that can move in both directions, which can read and write a character in the given field; the Turing machine is in a certain state (current state) at all times, which is one of a finite number of states; it can accept a set Turing program (program), which is a list of transitions that determines a new state for a given state and the character under the head, a character that must be written to the field under the head, and the direction of the head's movement, that is, left, right, or stationary.
- Components of a Turing Machine:
    1. Tape:
        - An infinite tape divided into cells.
        - Each cell contains a symbol from a finite alphabet (e.g., 0, 1, or a blank symbol □).
        - The tape serves as both input and unlimited memory storage.
    2. Head:
        - A movable read/write head that scans one cell of the tape at a time.
        - It can read a symbol, write a new symbol, and move left or right.
    3. Finite State Control:
        - A control unit with a finite number of states.
        - Determines the machine's actions based on the current state and the symbol under the head.
    4. Alphabet:
        - A set of symbols that the machine recognizes.
        - Includes a blank symbol (usually □) to represent empty cells on the tape.
    5. Transition Function:
        - A set of rules that dictate the machine's behavior.
        - Specifies what the machine should do (write a symbol, move the head, and change state) based on the current state and the symbol being read.
    6. Halting State:
        - A special state that signifies the machine has completed its computation.
- How a Turing Machine Works:
    1. Initialization:
        - The input is written on the tape, and the rest of the tape is filled with blank symbols.
        - The machine starts in an initial state with the head positioned y zat a designated starting point.
    2. Execution:
        - The machine reads the current symbol on the tape.
        - Based on the transition function, it:
            - Writes a new symbol (or leaves the current symbol unchanged).
            - Moves the head left or right.
            - Changes its state.
    3. Halting:
        - The machine stops when it enters a halting state or when no transition is defined for the current state and symbol.

binary
- 8 bit = 2^0 … 2^7
- base 10 → base n: divide by n, use reminder from bottom to top
    - decimal: times by n, use integer from top to bottom
- base n → base 10: times bit by the digit and add all together
- base x → base y: base x → base 10 → base y

storage
- bit & byte
- complement for negative
    - 1’s complement (求反)→ switch 1s and 0s
    - 2’s complement (补码)→ switch 1s and 0s after the first 1 from the right
    - transform back: 0 in left most digit → negative
- float
    - essentially just scientific notation in base 2
        - symbol (符号) + significant digit (定点数) + power to the base (位移量)
    - stored in computer: symbol (符号) + power (指数)+ significant digit not including the whole number (尾数)

computer
- Major Computer Subsystems:
    - CPU: Performs operations on data, consisting of:
        - Arithmetic Logic Unit (ALU; 算术逻辑单元): Handles arithmetic, shift, and logic operations.
        - Control Unit (控制单元): Manages CPU operations.
        - Registers (寄存器): Temporary, fast storage devices for data.
    - Main Memory (主存): Collection of storage cells identified by unique addresses.
        - Transfers data in binary bit groups (words).
        - Two types:
            - RAM: Random Access Memory (随机存取存储器).
            - ROM: Read-Only Memory (只读存储器).
        - Address Space (地址空间): Total number of uniquely identifiable cells.
    - Input/Output (I/O) Subsystems:
        - Connects computer to the external world.
        - Non-Storage Devices (非存储设备): Facilitate communication with the CPU/memory.
        - Storage Devices (存储设备): Store and retrieve data (e.g., magnetic and optical storage).
- Subsystem Connections:
    - Buses (总线):
        - Data Bus (数据总线): Transfers data.
        - Address Bus (地址总线): Transfers address.
        - Control Bus (控制总线): Manages control signals.
    - I/O Controllers/Interfaces: Devices like SCSI, FireWire, and USB connect I/O devices to buses.
- I/O Addressing Methods:
    - Independent Addressing (I/O独立寻址): Separate instructions for memory and I/O device operations.
    - Memory-Mapped Addressing (I/O存储器映射寻址): CPU treats I/O controller registers as memory words.
- Program Execution:
    - Programs: Instructions stored in memory along with data.
    - Machine Cycle (机器周期):
        - Instruction Fetch (取指令): Retrieve instruction.
        - Decode (译码): Interpret instruction.
        - Execute (执行): Perform operation.
- CPU and I/O Synchronization:
    - Program-Controlled I/O (程序控制I/O): CPU manages all I/O.
    - Interrupt-Controlled I/O (中断控I/O): Devices signal CPU to handle I/O.
    - Direct Memory Access (DMA; 直接存储器存取): Transfers data between memory and devices without CPU intervention.
- Computer Architectures:
    - CISC (复杂指令 集计算机): Complex Instruction Set Computers.
    - RISC (精简指令集计算机): Reduced Instruction Set Computers.
- Performance Enhancements:
    - Pipelining (流水线技术): Overlapping machine cycle stages to increase throughput.
    - Parallel Processing: Multiple instruction streams process data streams simultaneously.
- Modern Computer Trends:
    - Evolved architecture and organization.
    - Increased use of synchronization and optimization techniques.

logical operations
- not (~)
    - 0 → 1
    - 1 → 0
- and (&)
    - 0 and 0 → 0
    - 0 and 1 → 0
    - 1 and 0 → 0
    - 1 and 1 → 1
- or (|)
    - 0 or 0 → 0
    - 0 or 1 → 1
    - 1 or 0 → 1
    - 1 or 1 → 1
- xor (exclusive or; ^)
    - 0 xor 0 → 0
    - 0 xor 1 → 1
    - 1 xor 0 → 1
    - 1 xor 1 → 0

shift operations
- logical left/right → just shift and add 0 at right end (the leftmost disappears)
- circular left/right → leftmost comes to the right end
- arithmetic shift → Arithmetic shift operations assume that the bit pattern is a signed integer represented in two's complement format. Arithmetic right shift is used to divide integers by 2; and arithmetic left shift is used to multiply integers by 2. These operations should not change the sign bit (leftmost). Arithmetic right shift preserves the sign bit, but also copies it to the adjacent right bit, so the sign is preserved. Arithmetic left shift discards the sign bit and accepts the bit to its right as the sign bit. If the new sign bit is the same as the old one, the operation succeeds, otherwise overflow or underflow occurs and the result is illegal.

network
- 网络协议是指在计算机网络中，为了实现不同设备之间的通信而定义的一系列规则和约定。网络协议通常按照OSI（开放系统互联）参考模型或者TCP/IP模型的各个层次进行划分。以下是网络中各层常见的协议：
1. **物理层（Physical Layer）**
    - 物理层负责通过物理媒介传输原始的比特流。它的主要任务是定义电气信号、传输介质等。
    - 协议：物理层通常没有具体的“协议”定义，但涉及的标准有：
        - Ethernet (以太网)
        - DSL (数字用户线路)
        - 光纤传输协议（如Fiber Channel）
        - USB
        - Wi-Fi（也涉及到数据链路层，但物理层部分是无线通信标准）
2. **数据链路层（Data Link Layer）**
    - 数据链路层负责在物理链路上进行数据帧的传输与接收，保证数据的可靠性。
    - 协议：
        - **Ethernet（以太网）**：用于局域网中的数据传输。
        - **PPP（Point-to-Point Protocol，点对点协议）**：用于点对点链路，常用于拨号上网。
        - **Wi-Fi（无线局域网）**：用于无线通信。
        - **Frame Relay**：一种用于连接多个局域网的协议，较早期常见。
        - **HDLC（High-Level Data Link Control）**：一种面向比特的协议，用于帧的交换。
3. **网络层（Network Layer）**
    - 网络层负责数据包从源主机到目的主机的传输，主要处理路由、寻址等问题。
    - 协议：
        - **IP（Internet Protocol，互联网协议）**：主要用于在不同网络之间传输数据包，常见的版本有IPv4和IPv6。
        - **ICMP（Internet Control Message Protocol，互联网控制报文协议）**：用于发送控制消息，如ping命令。
        - **ARP（Address Resolution Protocol，地址解析协议）**：将IP地址解析为MAC地址。
        - **RARP（Reverse ARP）**：将MAC地址解析为IP地址（较少使用）。
        - **OSPF（Open Shortest Path First）**：一种动态路由协议，常用于大型网络中的路由选择。
        - **BGP（Border Gateway Protocol）**：用于互联网中的路由选择，支持跨自治系统的路由。
4. **传输层（Transport Layer）**
    - 传输层负责端到端的数据传输，确保数据的可靠性和顺序。
    - 协议：
        - **TCP（Transmission Control Protocol，传输控制协议）**：面向连接、可靠的传输协议，确保数据的顺序性和完整性。
        - **UDP（User Datagram Protocol，用户数据报协议）**：无连接、低延迟的传输协议，不保证可靠性。
        - **SCTP（Stream Control Transmission Protocol）**：面向消息的协议，结合了TCP和UDP的一些特点。
5. **会话层（Session Layer）**
    - 会话层负责建立、管理和终止会话。它确保不同计算机间的通信保持正确的顺序。
    - 协议：
        - **NetBIOS**：提供网络会话管理服务。
        - **RPC（Remote Procedure Call，远程过程调用）**：允许不同主机上的程序进行通信。
        - **SMB（Server Message Block）**：用于共享文件和打印机资源。
        - **TLS/SSL（Transport Layer Security / Secure Sockets Layer）**：用于在会话层加密数据传输。
6. **表示层（Presentation Layer）**
    - 表示层负责数据的格式化、加密和压缩，以便于应用层能够正确理解数据。
    - 协议：
        - **JPEG**、**GIF**、**PNG**：常见的图像文件格式。
        - **MPEG**、**MP3**：常见的音频和视频压缩格式。
        - **ASCII**、**EBCDIC**：字符编码格式。
        - **TLS/SSL**：可以在表示层加密数据。
7. **应用层（Application Layer）**
    - 应用层是与用户直接交互的层，负责提供应用程序需要的通信服务。
    - 协议：
        - **HTTP（Hypertext Transfer Protocol，超文本传输协议）**：用于网页浏览。
        - **HTTPS（Hypertext Transfer Protocol Secure）**：HTTP的安全版本，使用SSL/TLS进行加密。
        - **FTP（File Transfer Protocol，文件传输协议）**：用于文件传输。
        - **SMTP（Simple Mail Transfer Protocol，简单邮件传输协议）**：用于发送电子邮件。
        - **POP3（Post Office Protocol 3，邮局协议3）**：用于接收电子邮件。
        - **IMAP（Internet Message Access Protocol，互联网邮件访问协议）**：比POP3更强大的电子邮件接收协议。
        - **DNS（Domain Name System，域名系统）**：用于将域名解析为IP地址。
        - **Telnet**：用于远程登录服务器。
        - **SSH（Secure Shell）**：用于远程登录，具备加密功能。
        - **DHCP（Dynamic Host Configuration Protocol，动态主机配置协议）**：用于动态分配IP地址。
- 每一层都有其对应的协议和作用，它们共同合作使得网络通信得以顺畅进行。

**TCP/IP 四层模型**
1. **链路层 (Link Layer)**
    - 协议：与 OSI 数据链路层类似，包括 Ethernet, Wi-Fi 等。
2. **网络层 (Internet Layer)**
    - 协议：IP, ICMP, ARP (地址解析协议), RARP (逆向地址解析协议)
3. **传输层 (Transport Layer)**
    - 协议：TCP, UDP
4. **应用层 (Application Layer)**
    - 协议：包含了 OSI 模型中的会话层、表示层和应用层的功能，例如 HTTP, FTP, SMTP, DNS 等。
- **关键差异**
    - **OSI 模型** 更加理论化，将网络通信过程细分为七个层次。
    - **TCP/IP 模型** 是实际应用更广泛的模型，它简化为四个层次，更加注重实用性。
- 在实践中，大多数网络协议都是基于 TCP/IP 模型设计的，而 OSI 模型更多地用于教学目的和概念上的理解。然而，了解两者可以帮助更好地理解网络通信的工作原理。
- DNS
    - DNS 污染是指某些网络服务提供商或恶意软件篡改了 DNS 解析结果，将用户引导到错误的 IP 地址。
    - DNS 污染通常发生在以下几种情况下: 1.ISP 篡改:一些互联网服务提供商(ISP)可能会故意篡改 DNS 解析结果，将用户重定向到广告页面或其他网站。 2.恶意软件:某些恶意软件会修改本地的 hosts 文件或系统设置，以达到类似的目的。 3.政府控制:在某些国家和地区，政府可能会实施 DNS 污染来阻止访问特定网站。
    - **如何解决 DNS 污染** 使用公共 DNS 服务器 你可以尝试使用公共 DNS 服务器，如 Google Public DNS (8.8.8.8和 8.8.4.4)或 Cloudflare DNS (1.1.1.1 和1.0.0.1)，这些服务器通常不会被篡改。 更改 DNS 设置: 可以通过系统偏好设置中的网络设置来更改 DNS 服务器地址。
    - 修改本地 hosts 文件 如果你确定某个域名被污染，可以手动修改本地的 hosts 文件，将正确的IP 地址与域名绑定。
- **SSH**
    - SSH（Secure Shell）协议位于OSI七层模型的应用层。它主要用于在不安全的网络中为网络连接提供安全保障，特别是用于远程登录和执行命令。SSH不仅提供了强大的加密机制来保护数据的安全性和完整性，还支持身份验证，以确保通信双方的真实性。
    - 尽管SSH主要用于终端仿真和命令行界面的交互，它同样可以用来传输文件（通过SFTP或SCP）或者隧道其他应用程序。由于它的功能直接服务于应用程序之间的交互，并且是用户可以直接调用的服务，因此被归类到应用层。
    - 在TCP/IP模型中，SSH也是位于应用层。TCP/IP模型不像OSI模型那样严格区分表示层和会话层的功能，而是将这些功能都包含在了应用层中。所以，在这两个网络模型中，SSH都被认为是在应用层工作。
- type
1. **A 类网络 (Class A)**
    - **地址范围**：`1.0.0.0` 到 `126.255.255.255`
        - **二进制表示**：`00000001.00000000.00000000.00000000` 到 `01111110.11111111.11111111.11111111`
        - **十六进制表示**：`01.00.00.00` 到 `7E.FF.FF.FF`
    - **默认子网掩码**：`255.0.0.0` 或 `/8`
        - **二进制掩码**：`11111111.00000000.00000000.00000000`
        - **十六进制掩码**：`FF.00.00.00`
    - **主机部分大小**：A 类地址的主机部分占 **24 位**（32 位减去 8 位的网络部分）。
        - 可分配的主机地址数：
            
            ```
            2^24 - 2 = 16,777,214
            
            ```
            
            - 去掉网络地址和广播地址，剩余可分配给主机的地址数为 **16,777,214** 个。
2. **B 类网络 (Class B)**
    - **地址范围**：`128.0.0.0` 到 `191.255.255.255`
        - **二进制表示**：`10000000.00000000.00000000.00000000` 到 `10111111.11111111.11111111.11111111`
        - **十六进制表示**：`80.00.00.00` 到 `BF.FF.FF.FF`
    - **默认子网掩码**：`255.255.0.0` 或 `/16`
        - **二进制掩码**：`11111111.11111111.00000000.00000000`
        - **十六进制掩码**：`FF.FF.00.00`
    - **主机部分大小**：B 类地址的主机部分占 **16 位**（32 位减去 16 位的网络部分）。
        - 可分配的主机地址数：
            
            ```
            2^16 - 2 = 65,534
            
            ```
            
            - 去掉网络地址和广播地址，剩余可分配给主机的地址数为 **65,534** 个。
3. **C 类网络 (Class C)**
    - **地址范围**：`192.0.0.0` 到 `223.255.255.255`
        - **二进制表示**：`11000000.00000000.00000000.00000000` 到 `11011111.11111111.11111111.11111111`
        - **十六进制表示**：`C0.00.00.00` 到 `DF.FF.FF.FF`
    - **默认子网掩码**：`255.255.255.0` 或 `/24`
        - **二进制掩码**：`11111111.11111111.11111111.00000000`
        - **十六进制掩码**：`FF.FF.FF.00`
    - **主机部分大小**：C 类地址的主机部分占 **8 位**（32 位减去 24 位的网络部分）。
        - 可分配的主机地址数：
            
            ```
            2^8 - 2 = 254
            
            ```
            
            - 去掉网络地址和广播地址，剩余可分配给主机的地址数为 **254** 个。

other things to know / take note of
- important ppl
    - A.M Turing
    - John Von Neumann
    - Stephen A.Cook
- string[start:stop:step]
- 要反转二进制数01001001的高4位，保留低4位不变，应将它与11110000进行异或运算。
- 为了计算CPU的运算速度(以每秒百万条指令，即MIPS为单位)，我们可以使用以下公式: MIPS = 时钟频率 / 每个指令周期的时钟周期数
    - 给定条件是:
        - 时钟频率(主频)是4GHz，也就是4,000,000,000 Hz。
        - 每个指令周期需要8个时钟周期。
- 使用2个字节表示整数，则表示的范围可能是: -32768 ~ 32767
    - 2^16 = 65537 → allocate to both negative and positive
- “True” = non-zero integer
- email related: POP/SMT
- IPv6 & IPv4:
    
    
    | Feature | IPv4 | IPv6 |
    | --- | --- | --- |
    | Address Size | 32 bits (~4.3 billion) | 128 bits (~340 undecillion) |
    | Address Format | Decimal (e.g., 192.0.2.1) | Hexadecimal (e.g., 2001:db8::1) |
    | Header Size | 20 bytes | 40 bytes |
    | Security | Not built-in | Built-in IPsec |
    | Broadcast | Supported | Replaced with multicast |
    | Addressing | NAT required | No NAT needed |
- picture encoding
    - 以颜色编码为基础，将二维平面 **(**空间**)**离散化为网格点(像素 Pixel)，记录每个网格点上的一个代表性颜色值
    - 分辨率：网格点的数目，如1024x768
    - Space (Bytes)=Width (pixels)×Height (pixels)×Bits per Pixel (bpp)
        - 1MB=1,048,576Bytes (binary MB) or 1MB=1,000,000Bytes (decimal MB).
- video encoding
    - Video Space (Bytes)=Frame Size (Bytes)×Frame Rate (FPS)×Duration (Seconds)
        - RGB 每种颜色各 1 个字节
- audio encoding
    - time & wavelength → binary
    - Audio Space (Bytes)=Sample Rate (Hz)×Bit Depth (bits)×Channels×Duration (Seconds)÷8
- compress
    - Compressed File Size (Bytes)=Bitrate (bps)×Duration (Seconds)÷8
- disk storage
    - Disk Space Required (Bytes)=File Size (Bytes)+File System Overhead
        - for single file
- overflow (溢出)
- 加法 00000001
- 减法 00000010
- 乘法 00000011
- 除法 00000100
- 跳转 00000101
- 看给几个空。三个的话：中央处理器 (CPU)、存储器 (Memory)、输入/输出 (I/O)。五个的话：运算器 (Arithmetic Logic Unit, ALU)、控制器 (Control Unit)、存储器 (Memory)、输入设备 (Input Devices)、输出设备 (Output Devices)
    
    ![Screenshot 2025-01-04 at 11.30.19.png](computing%20written%20exam%20summary%20notes%2017086ecb9bf68005bfcee11617b3e9be/Screenshot_2025-01-04_at_11.30.19.png)